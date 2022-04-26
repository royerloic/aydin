from typing import Tuple

import numpy

from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class Dense(Module):
    def __init__(
        self, *args: Module, nb_inputs: int, nb_outputs: int, parameter_abs_limit=2
    ):
        """
        A dense modules is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        super().__init__(*args)
        self.nb_inputs: int = nb_inputs
        self.nb_outputs: int = nb_outputs
        self.parameter_abs_limit = parameter_abs_limit
        self._input: Tensor = None

        self.weights: Tensor = None
        self.biases: Tensor = None
        self._grad_weights: Tensor = None
        self._grad_biases: Tensor = None

    def ensure_parameters_allocated(self, x):
        """
        Ensures that the parameters of this module are allocated,
        uses x as a template (for class and dtype) to instantiate the tensors
        """
        if self.weights is None:
            self.weights: Tensor = x.new(shape=(self.nb_inputs, self.nb_outputs))
            std_dev: float = numpy.sqrt(2.0 / (self.nb_inputs + self.nb_outputs))
            self.weights.normal(mean=0.0, std_dev=std_dev)
            self._grad_weights = self.weights.new(
                shape=self.weights.shape, dtype=self.weights.dtype
            )

        if self.biases is None:
            self.biases: Tensor = x.new(shape=(self.nb_outputs,))
            self.biases.fill(0.0)
            self._grad_biases = self.biases.new(
                shape=self.biases.shape, dtype=self.biases.dtype
            )

    def __str__(self):
        return f"Dense [inputs:{self.nb_inputs}, outputs:{self.nb_outputs}]"

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return 'weights', 'biases'

    @property
    def parameters(self) -> Tuple[Tensor, ...]:
        return self.weights, self.biases

    @parameters.setter
    def parameters(self, parameters: Tuple[Tensor, ...]):
        self.weights = parameters[0]
        self.biases = parameters[1]

    @property
    def gradients(self):
        return self._grad_weights, self._grad_biases

    def forward(self, x: TensorTree) -> Tensor:
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        self.ensure_parameters_allocated(x)

        self._input = super()._children_forward(x)

        self._init_output(self._input, shape=(self._input.shape[0], self.nb_outputs))
        self._output.affine(self._input, self.weights, self.biases)

        return self._output

    def zero_gradients(self):
        self._grad_weights.fill(0.0)
        self._grad_biases.fill(0.0)
        super().zero_gradients()

    def backward(self, dy: Tensor) -> TensorTree:

        x = self._input

        batch_size = x.shape[0]
        self._init_gradient_outputs(x, (batch_size, self.nb_inputs))

        self._gradient_outputs[0].dot(dy, self.weights, tb=True)

        # compute gradient w.r.t. weights and biases
        self._grad_weights.dot(x, dy, ta=True, additive=True)
        self._grad_biases.sum(dy, axis=0)

        assert (
            self._grad_weights.shape == self.weights.shape
            and self._grad_biases.shape == self.biases.shape
        )

        return super()._children_backward(self._gradient_outputs)

    def constrain_parameters(self):
        """
        Normalisation consists in first removing weight bias, and then constraining biases and weights to [-l:1l].
        Assumption is that all inputs and outputs of the modules are ideally within [0, 1]
        """
        # Weights and biases should remain with reasonable bounds:
        l = self.parameter_abs_limit
        self.biases.clip(-l, l)
        self.weights.clip(-l, l)
