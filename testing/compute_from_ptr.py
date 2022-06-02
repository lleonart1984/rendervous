import rendering as ren
import torch

ren.create_device(device=0)
ren.compile_shader_sources('.')

# Create a torch tensor
x_torch = torch.rand(100, device=ren.torch_device())

# Create an output tensor
y_torch = torch.zeros_like(x_torch)

ptrs = ren.wrapped_ptr_collection(2)  # Creates a single uniform with capacity to store 2 GPU Ptrs

# automatically changes the underlying tensors to vulkan buffer and prepare the gpu address to the shader
ptrs.wrap(x_torch, y_torch)  # Copies the gpu ptrs to the uniform

pipeline = ren.pipeline_compute()
pipeline.load_shader('./compute_from_ptr.comp.spv')  # The extension is changed to spv compiled.
pipeline.bind_wrap_gpu_collection(0, lambda: ptrs)  # Bind the single uniform with the 2 ptrs
pipeline.bind_constants(0, dim=int)
pipeline.close()

with ren.compute_manager() as man:
    man.set_pipeline(pipeline)
    man.update_sets(0)
    man.update_constants(ren.ShaderStage.COMPUTE, dim=100)
    man.dispatch_threads_1D(100)

print(y_torch)

ren.quit()

