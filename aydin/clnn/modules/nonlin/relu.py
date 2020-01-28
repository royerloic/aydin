from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class ReLU(Module):
    def __init__(
        self, *args: Module, residual_gradient: float = 1e-5, zero_grad: bool = True
    ):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        """
        super().__init__(*args)
        self.zero_grad = zero_grad
        self.residual_gradient = residual_gradient
        self._input = None

    def forward(self, x: TensorTree) -> Tensor:
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        self._input = super()._children_forward(x)
        self._init_output(self._input, shape=self._input.shape)
        self._output.relu(self._input)
        return self._output

    def backward(self, dy: Tensor) -> TensorTree:
        """Compute gradient of loss w.r.t. ReLU input"""
        x = self._input
        self._init_gradient_outputs(x, x.shape)
        if self.zero_grad:
            self._gradient_outputs[0].signum_select(
                x, dy, dy, 1.0, -self.residual_gradient
            )
        else:
            self._gradient_outputs[0].signum_select(x, dy, dy, 1.0, -1.0)
        return super()._children_backward(self._gradient_outputs)
