# include <stdio.h>
# include "./arena.c"

typedef struct{
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


int main(void){
    int c;
    FILE *fp = fopen("./gpt2.bin","rb");
    char magic[4];
    fread(magic,1,4,fp);
    memcmp(magic, "CGPT", 4) == 0;
    Model model;

    // The below code could also be done by reading 7 ints into the cfg address
    // I opted for this method instead of clarity and to make it easier if the parameter order needed to be changed
    model.cfg.n_layers = read_32_bit_int(fp);
    model.cfg.n_heads = read_32_bit_int(fp);
    model.cfg.d_model = read_32_bit_int(fp);
    model.cfg.vocab = read_32_bit_int(fp);
    model.cfg.context = read_32_bit_int(fp);
    model.cfg.d_ff = read_32_bit_int(fp);
    Arena *a;
    arena_init(a, 600 * 1024 * 1024);
    // Token embedding
    model.wte = load_tensor(fp, a, model.cfg.vocab * model.cfg.d_model);
    // Positional embedding
    model.wpe =  load_tensor(fp, a, model.cfg.context * model.cfg.d_model);

    printf("%f\n", model.wte[0]);

}

int read_32bit_int(fp){
    int return_val;
    fread(&return_val, sizeof(int), 1, fp);
    return return_val;
}
// In C we do not have a way to structure tensors
// We just load contiguous blocks of values and then mentally view them as tensors
float * load_tensor(FILE *fp, Arena *a, size_t n){
    float *p = arena_alloc(a, n*sizeof(float));
    fread(p, sizeof(float),n,fp);
    return p;
}

