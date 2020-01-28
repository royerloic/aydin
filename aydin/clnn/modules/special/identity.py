from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class Identity(Module):
    def __init__(self, *args: Module):
        """
        A module that returns its input identically.
        """
        super().__init__(*args)
        self._input: Tensor = None
        self._output: Tensor = None

    def forward(self, x: TensorTree) -> Tensor:
        """
        Copies input to output.
        """
        self._input = super()._children_forward(x)
        self._init_output(self._input, shape=self._input.shape)
        self._output.copy_from(self._input, pad_value=0)
        return self._output

    def backward(self, dy: Tensor) -> TensorTree:

        x = self._input
        self._init_gradient_outputs(x, x.shape)
        self._gradient_outputs[0].copy_from(dy)
        return super()._children_backward(self._gradient_outputs)
