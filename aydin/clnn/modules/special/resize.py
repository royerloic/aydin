from typing import Tuple

from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class Resize(Module):
    def __init__(self, *args: Module, shape: Tuple[int]):
        """
        A module that pads or crops its input to match a given new tensor shape.
        """
        super().__init__(*args)
        self.shape = shape
        self._input: Tensor = None
        self._output: Tensor = None

    def __str__(self):
        return f"Resize[shape={self.shape}]"

    def forward(self, x: TensorTree) -> Tensor:
        """
        Resizes the input so that the output has a new given shape.
        """
        self._input = super()._children_forward(x)
        batch_size: int = self._input.shape[0]
        self._init_output(self._input, shape=(batch_size,) + self.shape)
        self._output.copy_from(self._input, pad_value=0)

        return self._output

    def backward(self, dy: Tensor) -> TensorTree:

        x = self._input
        self._init_gradient_outputs(x, x.shape)
        self._gradient_outputs[0].copy_from(dy)

        return super()._children_backward(self._gradient_outputs)
