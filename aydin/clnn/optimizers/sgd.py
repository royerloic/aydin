from aydin.clnn.modules.module import Module
from aydin.clnn.optimizers.optimizer import Optimizer
from aydin.clnn.tensor.tensor import Tensor


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimiser
    """

    def __init__(
        self,
        module: Module,
        learning_rate: float = Optimizer.LEARNING_RATE,
        momentum: float = 0.1,
        constrain_parameters: bool = False,
        normalise_gradients: bool = False,
        parameter_decay=Optimizer.DECAY,
        **kwargs,
    ):
        """
        Constructs a SGD optimiser.
        """
        super().__init__(
            module,
            learning_rate=learning_rate,
            parameter_decay=parameter_decay,
            constrain_parameters=constrain_parameters,
        )
        self.normalise_gradients = normalise_gradients
        self.momentum = momentum

    def reset(self, module: Module = None):
        super().reset(module)
        self._reset_for_postfix(module, '_delta')

    def optimise_parametric_module(self, module: Module):
        """
        Implements Stochastic Gradient Descent (SGD)
        """
        for name, param, grad in zip(
            module.parameter_names, module.parameters, module.gradients
        ):
            # Here we perform a stochastic gradient descent step.
            assert param.shape == grad.shape

            # Checking if there is a local scaling of the learning rate for this module.
            learning_rate = self.learning_rate
            if hasattr(module, '_learning_rate'):
                learning_rate *= module._learning_rate

            # Compute the parameter correction delta:

            delta = self._get_working_tensor(module, name, param, '_delta')

            if self.normalise_gradients:
                norm_grad = self._get_working_tensor(module, name, param, '_norm_grad')
                norm_grad.normalise(grad)
                delta.generalised_sum(
                    delta,
                    norm_grad,
                    self.momentum,
                    (1.0 - self.momentum) * learning_rate,
                )
            else:
                delta.generalised_sum(
                    delta, grad, self.momentum, (1.0 - self.momentum) * learning_rate
                )

            # Apply the parameter optimisation:
            param -= delta
