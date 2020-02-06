import numpy

from aydin.clnn.modules.linear.dense import Dense
from aydin.clnn.modules.losses.l2 import L2Loss
from aydin.clnn.modules.nonlin.abs import Abs
from aydin.clnn.optimizers.adam import ADAM
from aydin.clnn.optimizers.sgd import SGD
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor
from aydin.util.log.log import lsection, Log, lprint


def test_basic_training():
    max_epochs = 10

    try:
        adam_cl_loss = basic_training_with(CLTensor, ADAM, max_epochs=max_epochs)
        sgd_cl_loss = basic_training_with(CLTensor, SGD, max_epochs=max_epochs)
        print(f"Losses: adam:{adam_cl_loss}, sgd:{sgd_cl_loss}")

    except KeyboardInterrupt:
        pass

    try:
        adam_np_loss = basic_training_with(NPTensor, ADAM, max_epochs=max_epochs)
        sgd_np_loss = basic_training_with(NPTensor, SGD, max_epochs=max_epochs)
        print(f"Losses: adam:{adam_np_loss}, sgd:{sgd_np_loss}")
    except KeyboardInterrupt:
        pass

    print(f"SUMMARY:")
    print(f"Losses: adam:{adam_cl_loss}, sgd:{sgd_cl_loss}")
    print(f"Losses: adam:{adam_np_loss}, sgd:{sgd_np_loss}")


def basic_training_with(
    tensor_class, optimiser_class, max_epochs=10, length_training_data=128000
):

    # Addresses bug:
    CLTensor.opencl_provider = None

    Log.override_test_exclusion = False
    Log.set_log_max_depth(4)

    width = 16
    module = None
    module = Abs(Dense(module, nb_inputs=2, nb_outputs=width))
    module = Abs(Dense(module, nb_inputs=width, nb_outputs=width))
    module = Abs(Dense(module, nb_inputs=width, nb_outputs=width))
    module = Dense(module, nb_inputs=width, nb_outputs=1)

    loss = L2Loss(module)

    lprint(loss.tree())

    lprint(f"Tensor class: {tensor_class}")
    lprint(f"Optimiser class: {optimiser_class}")
    optimiser = optimiser_class(loss, learning_rate=1e-4)

    min_loss = 1

    a_np = numpy.random.random((length_training_data,)).astype(numpy.float32)
    b_np = numpy.random.random((length_training_data,)).astype(numpy.float32)
    x_np_all = numpy.stack((a_np, b_np), axis=-1)
    y_np_all = (a_np * (b_np + 0.3)) ** 0.5 - 0.1
    y_np_all = y_np_all[..., numpy.newaxis]

    x_all = tensor_class(x_np_all)
    y_all = tensor_class(y_np_all)

    mini_batch_size = 1280

    x = tensor_class.instanciate(shape=(mini_batch_size, 2), dtype=numpy.float32)
    y = tensor_class.instanciate(shape=(mini_batch_size, 1), dtype=numpy.float32)

    optimiser.reset()
    for e in range(max_epochs):
        with lsection(f"Training main iteration: {e} ."):
            for i in range(1000):
                rnd_slice = numpy.random.choice(
                    x_np_all.shape[0], size=mini_batch_size, replace=True
                )
                seed = 1 + i + 100 * e
                x.sample(x_all, seed)
                y.sample(y_all, seed)

                # with lsection(f"Loss, inference, backprop, and optimise..."):
                l = loss(x, y)
                y_inf = loss.actual  # module(x)

                loss.zero_gradients()
                loss.backprop()

                # print(modules)

                optimiser.step()

                # if i%100==0:
                #    lprint(f"{i}")

            min_loss = min(min_loss, l.nparray.mean())
            lprint(
                f"lr={optimiser.learning_rate}, actual={y_inf.nparray[0]}, target={y.nparray[0]}, loss={l.nparray.mean()}, min_loss={min_loss}"
            )
            optimiser.learning_rate *= 0.9

    loss_result = abs(l.nparray.mean())

    assert loss_result < 0.01

    return loss_result

    # loss = network([[0.5, 0.5],[1.0, 1.0]], -2)
    # print(loss)
