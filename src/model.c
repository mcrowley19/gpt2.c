# include <stdio.h>
# include "./arena.c"
# include <string.h>
typedef struct{
    int version;
    int n_layers;
    int n_heads;
    int d_model;
    int vocab;
    int context;
    int d_ff;
} Config;

typedef struct{
float *ln_1_w;
float *ln_1_bias;
float *attn_weight;
float *attn_bias;
float *attn_proj_weight;
float *attn_proj_bias;
float *ln_2_weight;
float *ln_2_bias;
float *fc_weight;
float *fc_bias;
float *mlp_proj_weight;
float *mlp_proj_bias;

} Block;

typedef struct{
    Config cfg;
    Block blocks[12];
    float *wte;
    float *wpe;
    float *ln_f_weight;
    float *ln_f_bias; 

} Model;

int read_32bit_int(FILE *fp){
    int return_val;
    fread(&return_val, sizeof(int), 1, fp);
    return return_val;
}

void check_parameter(const char* name, float *p, size_t n){
    double total = 0;
    for(size_t i = 0; i < n; i++){
        total = total + p[i];
    }

    printf("%s %.6f %.6f %.6f\n", name,p[0],p[n-1], total);
}


// In C we do not have a way to structure tensors
// We just load contiguous blocks of values and then mentally view them as tensors
float * load_tensor(FILE *fp, Arena *a, size_t n, const char *name){
    float *p = arena_alloc(a, n*sizeof(float));
    fread(p, sizeof(float),n,fp);
    check_parameter(name, p, n);
    return p;
}



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
    arena_free(&a);
    fclose(fp);
    return 0;
}
