#version 460

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0, std430) readonly buffer Input {
    float data[];
} _in;

layout(set=0, binding=1, std430) buffer Output {
    float data[];
} _out;

layout ( push_constant ) uniform Constants {
    int dim;
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    if (index >= dim)
    return;
    _out.data[index] = _in.data[index] + 2;
}
