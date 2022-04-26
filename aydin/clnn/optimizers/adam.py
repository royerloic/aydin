from aydin.clnn.modules.module import Module
from aydin.clnn.optimizers.optimizer import Optimizer


class ADAM(Optimizer):
    """
    Stochastic Gradient Descent Optimiser by ADAptive Moment Estimation
    """

    def __init__(
        self,
        module: Module,
        learning_rate: float = Optimizer.LEARNING_RATE,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        constrain_parameters: bool = False,
        parameter_decay=Optimizer.DECAY,
        **kwargs,
    ):
        """
        Constructs an ADAM optimiser.
        """
        super().__init__(
            module,
            learning_rate=learning_rate,
            constrain_parameters=constrain_parameters,
            parameter_decay=parameter_decay,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def reset(self, module: Module = None):
        super().reset(module)
        super()._reset_for_postfix(module, '_delta')
        super()._reset_for_postfix(module, '_mean')
        super()._reset_for_postfix(module, '_variance')

    def optimise_parametric_module(self, module: Module):
        """
        Implements ADAM parameter update
        """
        for name, param, grad in zip(
            module.parameter_names, module.parameters, module.gradients
        ):
            # Here we perform a stochastic gradient descent step.
            assert param.shape == grad.shape

            learning_rate = self.learning_rate
            if hasattr(module, '_learning_rate'):
                learning_rate *= module._learning_rate

            # Computes estimates for mean and variance:
            mean = self._get_working_tensor(module, name, param, '_mean')
            variance = self._get_working_tensor(module, name, param, '_variance')

            # if grad.has_nan_or_inf():
            #    raise Exception("NaN introduced during update!")

            # if mean.has_nan_or_inf() or variance.has_nan_or_inf():
            #    raise Exception("NaN introduced during update!")

            # This is not in the original ADAM, this seems to be a better initialisation...
            if self.optimisation_step_counter == 0:
                mean.generalised_sum(mean, grad, 0.0, 1.0)
                variance.generalised_sum(variance, grad, 0.0, 1.0, pb=2)
            else:
                mean.generalised_sum(mean, grad, self.beta1, (1.0 - self.beta1))
                variance.generalised_sum(
                    variance, grad, self.beta2, (1.0 - self.beta2), pb=2
                )

            # if mean.has_nan_or_inf() or variance.has_nan_or_inf():
            #    raise Exception("NaN introduced during update!")

            ## Computes and applies ADAM bias correction:
            # --> Turns out that this does not work well in practice...
            # Better to not initialise the mean and variance with zeros as done above...

            # t: int = self.optimisation_step_counter + 1
            # mean_bias_correction: float = 1.0 / (1.0 - self.beta1 ** t)
            # variance_bias_correction: float = 1.0 / (1.0 - self.beta2 ** t)
            # mean *= mean_bias_correction
            # variance *= variance_bias_correction

            # Computes ADAM delta:
            delta = self._get_working_tensor(module, name, param, '_delta')
            delta.generalised_product(
                mean,
                variance,
                sa=learning_rate,
                pb=0.5,
                ob=self.epsilon,
                mode='division',
            )

            # Apply the parameter optimisation:
            param -= delta

            # if param.has_nan_or_inf():
            #    raise Exception("NaN introduced during update!")
