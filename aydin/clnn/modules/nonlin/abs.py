from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class Abs(Module):
    def __init__(self, *args: Module):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        """
        super().__init__(*args)
        self._input = None

    def forward(self, x: TensorTree) -> Tensor:
        """Apply elementwise Abs to [batch, input_units] matrix"""
        self._input = super()._children_forward(x)
        self._init_output(self._input, shape=self._input.shape)
        self._output.abs(self._input)
        return self._output

    def backward(self, dy: Tensor) -> TensorTree:
        """Compute gradient of loss w.r.t. Abs input"""
        x = self._input
        self._init_gradient_outputs(x, x.shape)
        self._gradient_outputs[0].signum_select(x, dy, dy, 1.0, -1.0)
        return super()._children_backward(self._gradient_outputs)
