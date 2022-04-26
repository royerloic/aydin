from typing import Tuple

import numpy

from aydin.clnn.modules.losses.loss import LossModule
from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class L1Loss(LossModule):
    """ """

    def __init__(self, *args: Module):
        """ """
        super().__init__(*args)
        pass

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        self._init_output(x, shape=x.shape)
        self._output.absolute_diff(x, y)
        return self._output

    def loss_grad(self, x: Tensor, y: Tensor) -> TensorTree:
        self._init_gradient_outputs(x, x.shape)
        self._gradient_outputs[0].diff_sign(x, y)
        return tuple(self._gradient_outputs)
