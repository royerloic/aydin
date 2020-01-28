from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class LossModule(Module):
    """
    Loss modules are modules from which gradient originate.
    """

    def __init__(self, *args: Module):
        """

        """
        super().__init__(*args, nb_arguments=2)
        self._input: Tensor = None

    def __str__(self):
        return f"Loss"

    @property
    def actual(self) -> Tensor:
        return self._input[0]

    def loss(self, x: Tensor, target: Tensor) -> Tensor:
        raise NotImplemented()

    def loss_grad(self, x: Tensor, target: Tensor) -> TensorTree:
        raise NotImplemented()

    def backprop(self) -> Tensor:
        return self.backward(Tensor())

    def forward(self, x: TensorTree) -> Tensor:
        self._input = super()._children_forward(x)
        x = self._input[0]
        y = self._input[1]
        return self.loss(x, y)

    def backward(self, dy: Tensor) -> TensorTree:
        x = self._input[0]
        y = self._input[1]
        loss_gradient = self.loss_grad(x, y)
        return super()._children_backward(loss_gradient)
