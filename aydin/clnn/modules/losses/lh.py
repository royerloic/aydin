from typing import Tuple

import numpy

from aydin.clnn.modules.losses.loss import LossModule
from aydin.clnn.modules.module import Module, TensorTree
from aydin.clnn.tensor.tensor import Tensor


class LHalfLoss(LossModule):
    """ """

    def __init__(self, *args: Module, gradient_limit: float = 1):
        """ """
        super().__init__(*args)
        self.gradient_limit = gradient_limit

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        self._init_output(x, shape=x.shape)
        self._output.power_diff(x, y, p=0.5)
        return self._output

    def loss_grad(self, x: Tensor, y: Tensor) -> TensorTree:

        self._init_gradient_outputs(x, x.shape)
        self._gradient_outputs[0].power_diff(x, y, p=-0.5, retain_sign=True, alpha=0.5)
        self._gradient_outputs[0].clip(-self.gradient_limit, self.gradient_limit)
        return tuple(self._gradient_outputs)
