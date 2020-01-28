from typing import List, Tuple

from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class Sum(Module):
    def __init__(self, *args: Module):
        """
        Module that takes the sum of n inputs
        """
        super().__init__(*args)

    def forward(self, x: TensorTree) -> Tensor:
        """
        Compute activations of all modules by applying them sequentially.
        Return a list of activations for each layer.
        """
        self._input = super()._children_forward(x)
        first_tensor_in_sum = self._input[0]
        self._init_output(first_tensor_in_sum, first_tensor_in_sum.shape)

        # Looping through each layer
        for tensor in self._input:
            self._output += tensor

        return self._output

    def backward(self, dy: Tensor) -> TensorTree:

        x = self._input
        first_tensor_in_sum = x[0]
        self._init_gradient_outputs(
            first_tensor_in_sum, *tuple(tensor.shape for tensor in x)
        )

        # gradient is same for all terms of the sum:
        for output in self._gradient_outputs:
            output.copy_from(dy)

        return super()._children_backward(self._gradient_outputs)
