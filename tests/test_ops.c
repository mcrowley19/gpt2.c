#include "../src/ops.c"
#include <assert.h>


int main(){

    float W[6] = {1,2,3,4,5,6};
    
    float x[3] = {1,0,2};
    float out[2];


    matvec(out, W, x, 3, 2);
    assert(fabsf(out[0] - 7) < 1e-4);
    assert(fabsf(get_mean(W, 6) - 3.5f) < 1e-4);
    assert(fabsf(get_var(W, 6, 3.5f) - 2.9167f) < 1e-3);


    float ones[3] = {1,1,1};
    float zeros[3] = {0,0,0};
    float x3[3] = {1,2,3};
    float ln_out[3];


    layernorm(ln_out, x3, ones, zeros, 3);
    float expected_res[3] = {-1.2247, 0, 1.2247};
    for(int i = 0; i < 3; i++){
        assert(fabsf(ln_out[i] - expected_res[i]) < 1e-3);
    }
   
    float expected_gelu_res[3] = {0.8412, 1.9546, 2.9964};
    gelu(x3,3);
    for(int i = 0; i < 3; i++){
        assert(fabsf(x3[i] - expected_gelu_res[i]) < 1e-3);
    }

    float softmax_input[3] = {1,2,3};
    softmax(softmax_input, 3);
    float expected_softmax_output[3] = {0.0900, 0.2447, 0.6652};
    for(int i = 0; i < 3; i++){
        assert(fabsf(softmax_input[i] - expected_softmax_output[i]) < 1e-3);
    }

    float tensor_a[3] = {1,2,3};
    float tensor_b[3] = {10,20,30}; 

    resid_add(tensor_a,tensor_b, 3);
    float expected_resid_add_output[3] = {11,22,33};
    for(int i=0; i < 3; i ++){
        assert(fabsf(tensor_a[i] - expected_resid_add_output[i]) < 1e-3);
    }
}
