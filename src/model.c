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



