# include "./model.c"
# include "./ops.c"
typedef struct{
    float *k_cache[12];
    int kv_filled;
    float *v_cache[12]; // Should be cfg.n_layers but it needs to be a constant in C

    // Scratch tensors
    float *x;
    float *qkv;
    float *attn_out;
    float *tmp;
    float *ff;
    float *scores;
    float *logits;
} State;


void init_state(State *s, Model *m, Arena *a){
    Config c = m->cfg;
    for(int i =0; i < c.n_layers; i++){
        s -> k_cache[i] = arena_alloc(a, c.d_model* c.context * sizeof(float));
        s -> v_cache[i] = arena_alloc(a, c.d_model* c.context * sizeof(float));
    }
    s -> x = arena_alloc(a, c.d_model*sizeof(float));
    s -> qkv= arena_alloc(a, 3*c.d_model*sizeof(float));
    s -> attn_out= arena_alloc(a,c.d_model*sizeof(float));
    s -> tmp= arena_alloc(a,c.d_model*sizeof(float));
    s -> ff= arena_alloc(a,c.d_ff*sizeof(float));
    s -> scores= arena_alloc(a,c.context*sizeof(float));
    s -> logits= arena_alloc(a,c.vocab*sizeof(float));
    s-> kv_filled = 0;
}


void attention(State *s, int l, int pos, int d, int n_heads){
        memset(s->attn_out, 0, d * sizeof(float));
        int head_width = d / n_heads;
        for(int h = 0; h < n_heads; h++){ 
            // For qkv, k_cache and v_cache. Each head gets 64 elements and each is stored one after the other
            for(int token = 0; token <= pos; token++ ){
                s->scores[token] = 0;
                for(int c = 0; c < head_width; c++){
                    s->scores[token] = s->scores[token] + s->k_cache[l][h*head_width + token*d + c] * s->qkv[h*head_width + c];
                }
                s->scores[token] = s->scores[token] / sqrt(head_width);
            }
            softmax(s->scores, pos+1);
            for(int token = 0; token <= pos; token++){
                for(int i = 0; i < head_width; i++){
                    float output = (s->v_cache[l][h*head_width + token*d + i] * s->scores[token]);
                    s->attn_out[h*head_width + i] = s->attn_out[h*head_width + i] + output;
                }
            }

        }

}

void mlp(State *s, Block block, int d   ){
    layernorm(s->tmp, s->x, block.ln_2_weight, block.ln_2_bias, d);
    matvec(s->ff, block.fc_weight, s->tmp, d, 4*d);
    resid_add(s->ff, block.fc_bias, 4*d);
    gelu(s->ff, 4*d);
    matvec(s->tmp, block.mlp_proj_weight, s->ff, 4*d, d);
    resid_add(s->tmp, block.mlp_proj_bias, d);
    resid_add(s->x,s->tmp,d);
}

void forward(Model *m, State *s, int token, int pos){
    Config cfg = m-> cfg;
    for(int i = 0; i < cfg.d_model; i++){
        s->x[i] = m->wte[(token*cfg.d_model)+i ] + m->wpe[(pos*cfg.d_model)+i ];
    }

    for(int l = 0; l < cfg.n_layers; l++){
        Block block = m->blocks[l];
        layernorm(s->tmp, s->x, block.ln_1_w, block.ln_1_bias, cfg.d_model); // GPT-2 does its norm before the attention blocks, unlike AIAYN
        // qkv is three rows, each of length 768. A q row, a k row and a v row
        matvec(s->qkv, block.attn_weight, s->tmp, cfg.d_model, 3*cfg.d_model); // c_attn includes the Q,K and V matrices.
        resid_add(s->qkv, block.attn_bias, 3*cfg.d_model);
        memcpy(s->k_cache[l] + pos*cfg.d_model,s->qkv + cfg.d_model,  cfg.d_model * sizeof(float)); // k_cache is an array of k_caches for each of the layers
        memcpy(s->v_cache[l] + pos*cfg.d_model,s->qkv+2*cfg.d_model,  cfg.d_model *sizeof(float));
        attention(s, l, pos, cfg.d_model, cfg.n_heads);
        matvec(s->tmp, block.attn_proj_weight, s->attn_out, cfg.d_model, cfg.d_model);
        resid_add(s->tmp, block.attn_proj_bias,  cfg.d_model);
        resid_add(s->x, s->tmp, cfg.d_model);
        mlp(s, block, cfg.d_model);
    }
    layernorm(s->tmp,s->x, m->ln_f_weight, m->ln_f_bias, cfg.d_model); 
    matvec(s->logits, m->wte, s->tmp, cfg.d_model, cfg.vocab); 
}