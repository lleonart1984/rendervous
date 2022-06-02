import torch
import torch.nn as nn
import typing


class RendererModule(torch.nn.Module):
    """
    Module that performs gradient-based operations on graphics or raytracing pipelines.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setup(*args, **kwargs)
        self.__parameter_to_trigger_differentiation = nn.Parameter(torch.Tensor([0.0]))

    def setup(self, *args, **kwargs):
        """
        When implemented, creates the pipelines and resources necessary for the rendezvous process.
        """
        pass

    def forward_render(self, input: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        """
        Computes the output given the parameters
        """
        pass

    def backward_render(self, input: typing.List[torch.Tensor], output_gradients: typing.List[torch.Tensor]) -> \
    typing.List[torch.Tensor]:
        """
        Computes the gradient of parameters given the original inputs and the gradients of outputs
        """
        return [None for _ in input]  # assume by default no differentiable options for input

    def forward(self, *args):
        outputs = AutogradRendererFunction.apply(*(list(args) + [self.__parameter_to_trigger_differentiation, self]))
        return outputs[0] if len(outputs) == 1 else outputs


class AutogradRendererFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        renderer: RendererModule
        args = list(args)
        renderer = args[-1]
        inputs = args[0:-2]  # to skip parameter to trigger differentiation in output
        ctx.renderer = renderer
        outputs = renderer.forward_render(inputs)
        ctx.inputs = inputs
        # Only for checking in-place operations in the backward
        ctx.save_for_backward(*filter(lambda t: isinstance(t, torch.Tensor), inputs))
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        list(ctx.saved_tensors)  # Just check for inplace operations in input tensors
        inputs = ctx.inputs
        renderer = ctx.renderer
        grad_outputs = list(args)  # get output gradients
        grad_inputs = renderer.backward_render(inputs, grad_outputs)
        return tuple(grad_inputs + [None, None])  # append None to refer to renderer object passed in forward


def torch_device():
    """
    Gets the torch device where vulkan is running
    """
    return torch.device('cuda:0')