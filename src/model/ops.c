#include <math.h>
const float GELU_C = 0.7978845608f; // Approximately the square root of 2/pi
// out: space for return tensor, x: tensor to be normalised, w: ln_1_w, b: ln_1_bias
float get_mean(float *tensor, int n){
    float total = 0;
    for(int i = 0; i < n; i++){
        total += tensor[i];
    }
    return total/n;
}

// σ² = (1/n) Σ_i (x_i − μ)²
float get_var(float *tensor, int n, float mean){
    float total = 0;
    for(int i = 0; i < n; i++){
        float diff = tensor[i] - mean;
        float item_variance = (diff*diff)/n; // Multiplying faster than built in pow method
        total += item_variance;
    }
    return total;
}

// LayerNorm: y_i = (x_i − μ) / √(σ² + ε) · w_i + b_i, with ε = 1e-5
void layernorm(float *out, float *x, float *w, float*b, int n){
    float mean = get_mean(x,n);
    float var = get_var(x,n,mean);
    float eps = 1e-5;
    for(int i = 0; i < n; i++){
        out[i] = ((x[i] - mean)/sqrt(var+eps)) * w[i] + b[i];
    }
}

// In an (AxB) matrix mutliplied with a (B) size vector, in_dim is B and out_dim is A
void matvec(float *out, float *m1, float*m2, int in_dim, int out_dim){
    for(int i = 0; i < out_dim; i++){
        float sum = 0;
        for(int j=0; j<in_dim; j++){
            sum += (m1[i*in_dim + j] * m2[j]);
        }
        out[i] = sum;
    }

}

// GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
void gelu(float *x, int n){
    for(int i = 0; i < n; i++){
       x[i] = 0.5 * x[i] * (1 + tanh(GELU_C* (x[i] + 0.044715 * (x[i]*x[i]*x[i]))));
    }
}

// softmax(x)_i = e^{x_i − m} / Σ_j e^{x_j − m}
void softmax(float *x, int n){
    float max = x[0];
    for(int i = 1; i < n; i++){
        if (x[i] > max){
            max = x[i];
        }
    }
    float exp_sum = 0;
    for(int i = 0; i < n; i++){
        x[i] = exp(x[i] - max);
        exp_sum += x[i];
    }
    for(int i = 0; i < n; i++){
        x[i] = x[i] / exp_sum;
    }

}

void resid_add(float *x, float*y, int n){
    for (int i = 0; i < n; i++ ){
        x[i] += y[i];
    }
}

