#include "./forward.c"
int main(void){
    FILE *fp = fopen("./gpt2.bin","rb");
    char magic[4];
    fread(magic,1,4,fp);
    // Returns if the first 4 values in the binary file are not CGPT
    if (memcmp(magic, "CGPT", 4) != 0){
        return 1;
    }
    Model model;

    // The below code could also be done by reading 7 ints into the cfg address
    // I opted for this method instead of clarity and to make it easier if the parameter order needed to be changed
    model.cfg.version = read_32bit_int(fp);
    model.cfg.n_layers = read_32bit_int(fp);
    model.cfg.n_heads = read_32bit_int(fp);
    model.cfg.d_model = read_32bit_int(fp);
    model.cfg.vocab = read_32bit_int(fp);
    model.cfg.context = read_32bit_int(fp);
    model.cfg.d_ff = read_32bit_int(fp);
    Arena a;
    arena_init(&a, 600 * 1024 * 1024);
    // Token embedding matrix
    model.wte = load_tensor(fp, &a, model.cfg.vocab * model.cfg.d_model, "wte");
    // Positional embedding matrix
    model.wpe =  load_tensor(fp, &a, model.cfg.context * model.cfg.d_model, "wpe");
    int d = model.cfg.d_model;
    for(int i = 0; i < model.cfg.n_layers; i++){
        Block new_block;
        new_block.ln_1_w = load_tensor(fp, &a, d, "ln_1_w");
        new_block.ln_1_bias = load_tensor(fp, &a,d, "ln_1_bias");
        new_block.attn_weight = load_tensor(fp, &a,3 * d * d, "attn_weight");
        new_block.attn_bias = load_tensor(fp, &a,3 * d, "attn_bias");
        new_block.attn_proj_weight = load_tensor(fp, &a,d * d, "attn_proj_weight");
        new_block.attn_proj_bias = load_tensor(fp, &a,d, "attn_proj_bias");
        new_block.ln_2_weight = load_tensor(fp, &a,d, "ln_2_weight");
        new_block.ln_2_bias = load_tensor(fp, &a,d, "ln_2_bias");
        new_block.fc_weight = load_tensor(fp, &a,model.cfg.d_ff * d, "fc_weight");
        new_block.fc_bias = load_tensor(fp, &a,model.cfg.d_ff, "fc_bias");
        new_block.mlp_proj_weight = load_tensor(fp, &a,d * model.cfg.d_ff, "mlp_proj_weight");
        new_block.mlp_proj_bias = load_tensor(fp, &a,d, "mlp_proj_bias");
        model.blocks[i] = new_block;
    }

    model.ln_f_weight = load_tensor(fp, &a, d, "ln_f_weight");
    model.ln_f_bias =  load_tensor(fp, &a, d, "ln_f_bias");

    State s;
    init_state(&s, &model,&a);
    int tokens[] = {15496, 11, 616, 1438, 318};
    int n = sizeof(tokens) / sizeof(tokens[0]);

    for(int i = 0; i < n; i++){
        forward(&model, &s, tokens[i], i);
    }
    printf("%.6f\n", s.logits[0]);
    printf("%.6f\n", s.logits[model.cfg.vocab-1]);
    double res = 0;
    for(int i = 0; i < model.cfg.vocab; i++){
        res = res + s.logits[i];
    }
    printf("%.6f\n", res);

    arena_free(&a);
    fclose(fp);
    return 0;
}
