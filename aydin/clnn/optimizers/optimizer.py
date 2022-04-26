from aydin.clnn.modules.module import Module
from aydin.clnn.tensor.tensor import Tensor


class Optimizer:
    """
    Optimizer base class
    """

    LEARNING_RATE = 0.001
    DECAY = LEARNING_RATE * 1e-2

    def __init__(
        self,
        module: Module,
        learning_rate: float = LEARNING_RATE,
        parameter_decay: float = DECAY,
        constrain_parameters: bool = False,
    ):
        """
        Optimizer base class constructor
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.parameter_decay = parameter_decay
        self.constrain_parameters = constrain_parameters
        self.module: Module = module
        self.optimisation_step_counter: int

    def optimise_parametric_module(self, module):
        """
        Must be implemented to specify how each module should be optimised.
        """
        raise NotImplemented("")

    def reset(self, module=None):
        """
        Resets all caches and temp variables  used by the optimizer
        """
        if module is None:
            module = self.module

        self.optimisation_step_counter = 0
        for module in Module.traverse(module):
            module._optimiser_data = {}

    def _reset_for_postfix(self, module, post_fix):
        if module is None:
            module = self.module

        for module in Module.traverse(module):
            for name in module.parameter_names:
                key = name + post_fix
                if key in module._optimiser_data:
                    delta = module._optimiser_data[key]
                    if delta is not None:
                        delta.fill(0.0)

    def step(self, module=None):
        """
        Optimisation: goes through each module and optimises its parameters.
        """

        if module is None:
            module = self.module

        # print("____________________________")
        for module in Module.traverse(module):
            # print(module)
            if module.parameters:
                self.optimise_parametric_module(module)
                # Apply parameter decay:
                self.apply_parameter_decay(module)

                # Apply parameter constrains:
                if self.constrain_parameters:
                    module.constrain_parameters()

        self.optimisation_step_counter += 1

    def apply_parameter_decay(self, module: Module):
        """
        Applies parameter decay to given module.
        """
        for param in module.parameters:
            # Note: parameter decay is scaled with learning rate, so that it os always weaker than the learning rate
            param *= 1.0 - self.parameter_decay

    def _get_working_tensor(
        self, module: Module, name: str, param: Tensor, post_fix: str
    ) -> Tensor:
        d_key = name + post_fix
        if not d_key in module._optimiser_data:
            delta = param.new()
            module._optimiser_data[d_key] = delta
        return module._optimiser_data[d_key]
