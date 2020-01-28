from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class Noise(Module):
    def __init__(self, *args: Module, noise_level: float = 0.0001):
        """
        A module that adds uniform noise to a tensor.
        """
        super().__init__(*args)
        self.noise_level = noise_level
        self._input: Tensor = None
        self._output: Tensor = None

    def __str__(self):
        return f"Resize[shape={self.shape}]"

    def forward(self, x: TensorTree) -> Tensor:
        """
        Adds uniform noise to a tensor.
        """
        self._input = super()._children_forward(x)
        self._init_output(self._input, shape=self._input.shape)
        self._output.noise(self._input, noise_level=self.noise_level)

        return self._output

    def backward(self, dy: Tensor) -> TensorTree:

        x = self._input
        self._init_gradient_outputs(x, x.shape)
        self._gradient_outputs[0].copy_from(dy)

        return super()._children_backward(self._gradient_outputs)
