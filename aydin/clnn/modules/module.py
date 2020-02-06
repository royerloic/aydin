import collections
import copy
from typing import Tuple, List, Union

from aydin.clnn.tensor.tensor import Tensor
from aydin.util.log.log import lprint

SavedState = collections.namedtuple('SavedState', 'state children')
TensorTree = Union[Tensor, Tuple[Union[Tensor, 'TensorTree'], ...]]


class Module:
    """
    A building block. Each modules is capable of performing two things:
    - Process input to get output:           output = layer.forward(input)
    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)
    Some modules also have learnable parameters which can be updated.
    """

    def __init__(self, *args: 'Module', nb_arguments: int = None):
        """
        Constructs a module.
        """

        # A dummy layer does nothing
        super().__init__()
        if nb_arguments is None:
            nb_arguments = len(args)
        self.child_modules: List[Module] = [None] * nb_arguments
        self.child_modules[0 : len(args)] = args[:]

        # This dictionary is used to store data used by the optimizer:
        self._optimiser_data = {}

        # Tensor holding the inputs, outputs for forward and backward computations:
        self._input: Tensor = None
        self._gradient_outputs: List[Tensor] = None
        self._output: Tensor = None

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()

    def tree(self) -> str:
        """
        Returns the module tree (computation graph)
        """
        return self.__class__.__name__ + (
            f"({','.join([(child.tree() if child is not None else '_') for child in self.child_modules ])})"
            if self.child_modules
            else "(_)"
        )

    @classmethod
    def traverse(cls, module):
        """
        Generic recursive machinery that traverses the module tree.
        """
        yield module
        # recurse through child modules:
        for child_module in module.child_modules:
            if child_module is not None:
                yield from cls.traverse(child_module)

    def to_tensor_class(self, tensor_class):
        """
        Returns a copy of the module in which all tensors have been changed to the given tensor class, and so, recursively across children.
        """
        copied_module = copy.copy(self)
        new_parameters = tuple(
            parameter.to_class(tensor_class) for parameter in self.parameters
        )
        copied_module.parameters = new_parameters
        copied_module.child_modules = list(
            [
                None if child is None else child.to_tensor_class(tensor_class)
                for child in copied_module.child_modules
            ]
        )
        return copied_module

    def count_parameters(self) -> int:
        """
        Returns the number of parameters in module and all children, recursively.
        """
        count: int = 0
        for parameter in self.parameters:
            count += parameter.size
        for child in self.child_modules:
            if child is not None:
                count += child.count_parameters()
        return count

    @property
    def state(self) -> SavedState:
        parameters_np = tuple(tensor.nparray for tensor in self.parameters)
        children_states = tuple(
            [
                (child.state if child is not None else None)
                for child in self.child_modules
            ]
        )
        return SavedState(state=parameters_np, children=children_states)

    @state.setter
    def state(self, saved_state: SavedState):

        if saved_state is None:
            return

        for parameter, parameter_saved_np in zip(self.parameters, saved_state.state):
            parameter.nparray = parameter_saved_np

        for child_module, children_state in zip(
            self.child_modules, saved_state.children
        ):
            if child_module is not None:
                child_module.state = children_state

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """
        Returns a tuple containing all modules parameter names.
        """
        return tuple()

    @property
    def parameters(self) -> Tuple[Tensor, ...]:
        """
        Returns a tuple containing all modules parameters.
        """
        return tuple()

    @parameters.setter
    def parameters(self, val: Tuple[Tensor, ...]):
        """
        Sets parameters to a given tuple.
        """

    @property
    def gradients(self) -> Tuple[Tensor, ...]:
        """
        Returns a tuple containing all modules parameter gradients.
        """
        return tuple()

    def _init_output(self, existing_tensor: Tensor, shape: Tuple[int, ...]):
        """
        Initialises tensors used as output of the module's forward pass.
        """
        if (
            self._output is not None
            and self._output.__class__ != existing_tensor.__class__
        ):
            self._output = self._output.to_class(existing_tensor.__class__)

        if self._output is None or self._output.shape != shape:
            del self._output
            self._output = existing_tensor.new(shape=shape)

    def _init_gradient_outputs(self, existing_tensor: Tensor, *args: Tuple[int, ...]):
        """
        Initialises tensors used as output of the module's backward pass.
        """
        if not self._gradient_outputs:
            self._gradient_outputs = [None] * len(args)
        i: int = 0
        for gradient_output, shape in zip(self._gradient_outputs, args):
            if (
                gradient_output is not None
                and gradient_output.__class__ != existing_tensor.__class__
            ):
                self._gradient_outputs[i] = gradient_output.to_class(
                    existing_tensor.__class__
                )
            if gradient_output is None or gradient_output.shape != shape:
                self._gradient_outputs[i] = existing_tensor.new(shape=shape)
            i += 1

    def _children_forward(self, x: TensorTree) -> TensorTree:
        """
        Computes forward pass for all children, recursively.
        """
        if self.child_modules and not isinstance(x, Tuple):
            # In this case there is only one child:
            if not self.child_modules[0]:
                # If the child is None, ther eis no child -- equivalent to identity module.
                return x
            else:
                # There is one child module, we delegate:
                return self.child_modules[0](x)
        elif self.child_modules:
            return tuple(
                (module(u) if module is not None else u)
                for module, u in zip(self.child_modules, x)
            )
        else:
            return x

    def _children_backward(self, dys: List[Tensor]) -> TensorTree:
        """
        Computes backward pass for all children, recursively
        """
        if self.child_modules and isinstance(dys, Tensor):
            return self.child_modules[0].backward(dys)
        elif self.child_modules:
            return tuple(
                (module.backward(dy) if module is not None else dy)
                for module, dy in zip(self.child_modules, dys)
            )
        else:
            return dys

    def __call__(self, *args: TensorTree) -> Tensor:
        # converts list of arguments to tuple... so that result is consistent with recursive type TensorTree.
        if len(args) == 1:
            x = args[0]
            return self.forward(x)
        else:
            return self.forward(args)

    def forward(self, x: TensorTree) -> Tensor:
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        """
        return x

    def backward(self, dy: Tensor) -> TensorTree:
        """
        Performs a backpropagation step through the layer, with respect to the given input.
        """
        return dy

    def zero_gradients(self):
        """
        Zeroes gradients -- typically needed before optimising.
        """
        for module in self.child_modules:
            if module is not None:
                module.zero_gradients()

    def constrain_parameters(self):
        """
        If needed, modules can define how to constrain their parameters after training.
        """
        pass

    def check_for_nans_and_infs(self, assert_no_nans_or_infs=True, source='root'):
        """
        Checks for nans and infs throughout the compute tree.
        NOTE: This is slow!!! be carefull.
        """
        source = self.__class__.__name__ + "<--" + source

        for name, parameter in zip(self.parameter_names, self.parameters):
            if (not parameter is None) and parameter.has_nan_or_inf():
                lprint(
                    f"Found nan or inf in parameter '{name}' at {source}: \\n {parameter}",
                    Warning,
                )
                if assert_no_nans_or_infs:
                    assert False

        if self._output and self._output.has_nan_or_inf():
            lprint(
                f"Found nan or inf in output at {source}: \\n {self._output}", Warning
            )
            if assert_no_nans_or_infs:
                assert False

        for name, gradient in zip(self.parameter_names, self.gradients):
            if (not gradient is None) and gradient.has_nan_or_inf():
                lprint(
                    f"Found nan or inf in parameter gradient '{name}'at {source}: \\n {gradient}",
                    Warning,
                )
                if assert_no_nans_or_infs:
                    assert False

        if self._gradient_outputs:
            for gradient_out in self._gradient_outputs:
                if gradient_out and gradient_out.has_nan_or_inf():
                    lprint(
                        f"Found nan or inf in gradient output at {source}: \\n{gradient_out}",
                        Warning,
                    )
                    if assert_no_nans_or_infs:
                        assert False

        # recurse through children...
        for module in self.child_modules:
            if module is not None:
                module.check_for_nans_and_infs(
                    assert_no_nans_or_infs=assert_no_nans_or_infs, source=source
                )

    def check_for_zeros_parameters(
        self, assert_no_zero_parameters=False, source='root'
    ):
        """
        Checks for zeros in parameters throughout the compute tree (i.e. is a parameter all zeros).
        NOTE: This is slow!!! be carefull.
        """
        source = self.__class__.__name__ + "<--" + source

        for name, parameter in zip(self.parameter_names, self.parameters):
            if (not parameter is None) and parameter.is_zero():
                lprint(
                    f"Found all-zero parameter '{name}' at {source}: \\n {parameter}",
                    Warning,
                )
                if assert_no_zero_parameters:
                    assert False

        for name, gradient in zip(self.parameter_names, self.gradients):
            if (not gradient is None) and gradient.is_zero():
                lprint(
                    f"Found all-zero in parameter gradient '{name}'at {source}: \\n {gradient}",
                    Warning,
                )
                if assert_no_zero_parameters:
                    assert False

        # recurse through children...
        for module in self.child_modules:
            if module is not None:
                module.check_for_zeros_parameters(
                    assert_no_zero_parameters=assert_no_zero_parameters, source=source
                )
