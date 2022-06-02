#version 460
#extension GL_GOOGLE_include_directive : require
#include "../rendering/torch/shaders/AD.h"

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0, std430) uniform Tensors {
    GPUPtr in_tensor;
    GPUPtr out_tensor;
};

// this macro enables other functions to load and save values from tensors
// by using the gpu addresses.
require_tensor(1);

layout ( push_constant ) uniform Constants {
    int dim;
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    if (index >= dim)
    return;
    float x[1];
    tensorLoad(in_tensor, index, x);
    float y[1];
    y[0] = x[0] + 2;
    tensorSave(out_tensor, index, y);
}
