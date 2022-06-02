import rendering as ren
import torch


ren.create_device(device=0)
ren.compile_shader_sources('.')


# Create a torch tensor
x_torch = torch.rand(100, device=ren.torch_device())

# Create an output vulkan buffer visible to torch
y_buffer = ren.tensor_like(x_torch)

pipeline = ren.pipeline_compute()
pipeline.load_shader('./compute.comp.spv')  # The extension is changed to spv compiled.
pipeline.bind_storage_buffer(0, lambda: ren.as_buffer(x_torch))  # automatically changes the underlying tensor to vulkan buffer
pipeline.bind_storage_buffer(1, lambda: y_buffer)
pipeline.bind_constants(0, dim=int)
pipeline.close()

with ren.compute_manager() as man:
    man.set_pipeline(pipeline)
    man.update_sets(0)
    man.update_constants(ren.ShaderStage.COMPUTE, dim=100)
    man.dispatch_threads_1D(100)

y_torch = y_buffer.as_tensor()  # Zero-copy transform from vulkan buffer to a tensor.

print(y_torch)

ren.quit()

