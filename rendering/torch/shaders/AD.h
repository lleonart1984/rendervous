#ifndef AD_ACTIVATIONS_H
#define AD_ACTIVATIONS_H

//#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_shader_atomic_float : require

#define GPUPtr uint64_t

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatPointer { float data; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatArrayPointer { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer IntArrayPointer { int data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UintArrayPointer { uint data[]; };

float get_param(GPUPtr ptr, int index) { return FloatPointer(ptr + index * 4).data; }
float get_param(GPUPtr ptr) { return FloatPointer(ptr).data; }
void update_param(GPUPtr ptr, float de_dp) { atomicAdd(FloatPointer(ptr).data, de_dp); }
void update_param(GPUPtr ptr, int index, float de_dp) { atomicAdd(FloatPointer(ptr + index * 4).data, de_dp); }
//void update_param(uint64_t ptr, int index, float de_dp) { FloatPointer(ptr + index * 4).data += de_dp; }

float relu(float x){
    return x < 0 ? 0 : x;
}

void relu_bw(float x, float de_dy, out float de_dx)
{
    float dy_dx = x < 0 ? 0 : 1;
    de_dx = de_dy * dy_dx;
}

float leaky_relu(float x){
    return x < 0 ? 0.01 * x : x;
}

void leaky_relu_bw(float x, float de_dy, out float de_dx)
{
    float dy_dx = x < 0 ? 0.01 : 1;
    de_dx = de_dy * dy_dx;
}

float sigmoid(float x){
    return 1.0 / (1 + exp(-x));
}

void sigmoid_bw(float x, float de_dy, out float de_dx){
    float s = sigmoid(x);
    de_dx = s*(1 - s)*de_dy;
}

float sinact(float x){
    return sin(x * 3.141593 * 2);
}

void sinact_bw(float x, float de_dy, out float de_dx)
{
    float dy_dx = cos(x* 3.141593 * 2) * 3.141593 * 2;
    de_dx = de_dy * dy_dx;
}

float snake(float x){
    return x*0.5 + pow(sin(x), 2);
}

void snake_bw(float x, float de_dy, out float de_dx){
    float s = sin(2*x);
    de_dx = de_dy*(0.5 + s);
}

float elu(float x){
    return x >= 0 ? x : exp(x) - 1;
}

void elu_bw(float x, float de_dy, out float de_dx){
    float dy_dx = x >= 0 ? 1 : exp(x);
    de_dx = de_dy * dy_dx;
}

#define require_parameterless_activation(size, fname) \
void fname(in float x[(size)], out float y[(size)]) { for (int i=0; i<size; i++) y[i] = fname(x[i]); }\
void fname##_bw(in float x[size], in float de_dy[size], out float de_dx[size]) { for (int i=0; i<size; i++) fname##_bw(x[i], de_dy[i], de_dx[i]); }

#define require_relu(size) require_parameterless_activation(size, relu)
#define require_leaky_relu(size) require_parameterless_activation(size, leaky_relu)
#define require_sigmoid(size) require_parameterless_activation(size, sigmoid)
#define require_snake(size) require_parameterless_activation(size, snake)
#define require_elu(size) require_parameterless_activation(size, elu)
#define require_sinact(size) require_parameterless_activation(size, sinact)

#define require_load(size) \
void tensorLoad(GPUPtr ptr, int index, out float x[size]) { \
    FloatArrayPointer x_buf = FloatArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) x[i] = x_buf.data[i]; \
} \
void tensorLoadAdd(GPUPtr ptr, int index, float weight, inout float x[size]) { \
    FloatArrayPointer x_buf = FloatArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) x[i] += weight * x_buf.data[i]; \
} \
void tensorLoad(GPUPtr ptr, int index, out int x[size]) { \
    IntArrayPointer x_buf = IntArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) x[i] = x_buf.data[i]; \
}\
void tensorLoad(GPUPtr ptr, int index, out uint x[size]) { \
    UintArrayPointer x_buf = UintArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) x[i] = x_buf.data[i]; \
}

#define require_save(size) \
void tensorSave(GPUPtr ptr, int index, in float y[size]) { \
    FloatArrayPointer y_buf = FloatArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) y_buf.data[i] = y[i]; \
}\
void tensorSave(GPUPtr ptr, int index, in int y[size]) { \
    IntArrayPointer y_buf = IntArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) y_buf.data[i] = y[i]; \
}\
void tensorSave(GPUPtr ptr, int index, in uint y[size]) { \
    UintArrayPointer y_buf = UintArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) y_buf.data[i] = y[i]; \
}

#define require_add(size) \
void tensorAdd(GPUPtr ptr, int index, in float v[size]) { \
    FloatArrayPointer y_buf = FloatArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) y_buf.data[i] += v[i]; \
}\
void tensorAtomAdd(GPUPtr ptr, int index, in float dL_dx[size]) { \
    FloatArrayPointer y_buf = FloatArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) atomicAdd(y_buf.data[i], dL_dx[i]); \
}\
void tensorAtomAdd(GPUPtr ptr, int index, in float dL_dx[size], float w) { \
    FloatArrayPointer y_buf = FloatArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) atomicAdd(y_buf.data[i], dL_dx[i] * w); \
}\
void tensorAdd(GPUPtr ptr, int index, in int v[size]) { \
    IntArrayPointer y_buf = IntArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) y_buf.data[i] += v[i]; \
}\
void tensorAtomAdd(GPUPtr ptr, int index, in int dL_dx[size]) { \
    IntArrayPointer y_buf = IntArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) atomicAdd(y_buf.data[i], dL_dx[i]); \
}\
void tensorAdd(GPUPtr ptr, int index, in uint v[size]) { \
    UintArrayPointer y_buf = UintArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) y_buf.data[i] += v[i]; \
}\
void tensorAtomAdd(GPUPtr ptr, int index, in uint dL_dx[size]) { \
    UintArrayPointer y_buf = UintArrayPointer(ptr + index*4*size); \
    for (int i=0; i<size; i++) atomicAdd(y_buf.data[i], dL_dx[i]); \
}

#define require_tensor(size) require_load(size) require_save(size) require_add(size)

#define require_fourier_transform(in_features, L) \
void fourier_transform(in float x[in_features], out float y[in_features*(2*L)], float scale) { \
    int c = 0; \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) y[c++] = cos(scale * x[i] * 3.141593 * (1 << k)); \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) y[c++] = sin(scale * x[i] * 3.141593 * (1 << k)); \
} \
void fourier_transform_bw(in float x[in_features], in float de_dy[in_features*(2*L)], out float de_dx[in_features], float scale) { \
    int c = 0; \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) de_dx[i] -= sin(x[i] * scale * 3.141593 * (1 << k)) * (1 << k) * 3.141593 * scale * de_dy[c++]; \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) de_dx[i] += cos(x[i] * scale * 3.141593 * (1 << k)) * (1 << k) * 3.141593 * scale * de_dy[c++]; \
}

#define require_fourier_features(in_features, L) \
void fourier_features_fw(in float x[in_features], out float y[in_features*(2 * L + 1)]) { \
    int c = 0; \
    for (int i=0; i<in_features; i++) y[c++] = x[i]; \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) y[c++] = cos(x[i] * 3.141593 * (1 << k)); \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) y[c++] = sin(x[i] * 3.141593 * (1 << k)); \
} \
void fourier_features_bw(in float x[in_features], in float de_dy[in_features*(2 * L + 1)], out float de_dx[in_features]) { \
    int c = 0; \
    for (int i=0; i<in_features; i++) de_dx[i] += de_dy[c++]; \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) de_dx[i] -= sin(x[i] * 3.141593 * (1 << k)) * (1 << k) * 3.141593 * de_dy[c++]; \
    for (int k=0; k<L; k++) for (int i=0; i<in_features; i++) de_dx[i] += cos(x[i] * 3.141593 * (1 << k)) * (1 << k) * 3.141593 * de_dy[c++]; \
}



#define require_concat(dim1, dim2) \
void concat_fw(in float x1[dim1], in float x2[dim2], out float y[dim1 + dim2]) {\
    for(int i = 0; i < dim1; i++) y[i] = x1[i]; \
    for(int i = 0; i < dim2; i++) y[i + dim1] = x2[i]; \
} \
void concat_bw(in float dL_dy[dim1 + dim2], out float dL_dx1[dim1], out float dL_dx2[dim2]) {\
    for(int i = 0; i < dim1; i++) dL_dx1[i] = dL_dy[i]; \
    for(int i = 0; i < dim2; i++) dL_dx2[i] = dL_dy[i + dim1]; \
}


#define require_linear(in_features, out_features) \
void linear(in float x[(in_features)], out float y[(out_features)], GPUPtr weights, GPUPtr bias) { \
    FloatArrayPointer weights_buffer = FloatArrayPointer(weights); \
    FloatArrayPointer bias_buffer = FloatArrayPointer(bias); \
    int cw = 0; \
    int cb = 0; \
    for (int i = 0; i < out_features; i++) { \
        y[i] = 0.0; \
        for (int j=0; j<in_features; j++) y[i] += x[j] * weights_buffer.data[cw++]; \
        y[i] += bias_buffer.data[cb++]; \
    } \
} \
void linear_bw(in float x[in_features], in float de_dy[out_features], out float de_dx[in_features], GPUPtr weights, GPUPtr bias, GPUPtr de_dweights, GPUPtr de_dbias) { \
    for (int j=0; j < in_features; j++) de_dx[j] = 0; \
    FloatArrayPointer weights_buffer = FloatArrayPointer(weights); \
    FloatArrayPointer dweight_buffer = FloatArrayPointer(de_dweights); \
    int N = out_features * in_features; \
    for (int c = 0; c < N; c++) { \
        int i = c / in_features; \
        int j = c % in_features; \
        de_dx[j] += de_dy[i] * weights_buffer.data[c]; \
        atomicAdd(dweight_buffer.data[c], de_dy[i] * x[j]); \
    } \
    FloatArrayPointer dbias_buffer = FloatArrayPointer(de_dbias); \
    for (int i=0; i<out_features; i++) atomicAdd(dbias_buffer.data[i], de_dy[i]); \
}

#endif