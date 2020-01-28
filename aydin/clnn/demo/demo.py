# import napari
import random

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.clnn.modules.linear.dense import Dense
from aydin.clnn.modules.losses.l1 import L1Loss
from aydin.clnn.modules.nonlin.relu import ReLU
from aydin.clnn.modules.special.resize import Resize
from aydin.clnn.optimizers.adam import ADAM
from aydin.clnn.optimizers.optimizer import Optimizer
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor
from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.util.log.log import lsection


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def denoise_image(
    tensor_class=NPTensor,
    optimizer_class=ADAM,
    nb_layers=6,
    layer_width=None,
    activation_class=ReLU,
    loss_function_class=L1Loss,
):

    image = camera().astype(numpy.float32)
    image = n(image)
    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)
    generator = FastMultiscaleConvolutionalFeatures(
        max_level=10, exclude_scale_one=False
    )
    features = generator.compute(
        noisy, exclude_center_value=True, exclude_center_feature=True
    )
    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)[:, numpy.newaxis]

    nb_features = x.shape[-1]
    if layer_width is None:
        layer_width = nb_features
    dense_layers = []
    layers = []
    modules = []
    losses = []
    current_layer = None
    for i in range(nb_layers):
        dense = Dense(
            current_layer,
            nb_inputs=nb_features if i == 0 else layer_width,
            nb_outputs=layer_width,
        )
        dense._learning_rate = 1.0
        dense.ensure_parameters_allocated(tensor_class(0.0))
        if i > 0:
            weights = dense.weights.nparray
            weights[0, 0] = 1.0
            dense.weights.nparray = weights
        dense_layers.append(dense)
        act = activation_class(dense)
        layers.append(act)
        current_layer = act

        crop = Resize(act, shape=(1,))
        modules.append(crop)

        loss = loss_function_class(crop)
        losses.append(loss)

    x_t = tensor_class(x)
    y_t = tensor_class(y)
    for loss in losses:
        print(loss.tree())
    # Optimiser
    optimiser: Optimizer = optimizer_class(losses[-1], learning_rate=1e-3)
    best_val_loss_values = []
    best_state = [None] * nb_layers
    # Training happens here:
    try:
        optimiser.reset()
        mini_batch_size = 1024
        x_batch = tensor_class.instanciate(
            shape=(mini_batch_size, nb_features), dtype=numpy.float32
        )
        y_batch = tensor_class.instanciate(
            shape=(mini_batch_size, 1), dtype=numpy.float32
        )
        for e in range(20):
            print("")

            # make sure we start an epoch with the best possible state:
            for i in reversed(range(nb_layers)):
                if best_state[i]:
                    dense_layers[i].state = best_state[i]
            with lsection(f"Training main iteration: {e} ."):
                for i in range(256):

                    seed = random.randint(0, 2 ** 30)
                    x_batch.sample(x_t, seed)
                    y_batch.sample(y_t, seed)

                    loss_results = []
                    for loss in losses:
                        loss_result = loss(x_batch, y_batch)
                        loss_results.append(loss_result)

                    for i in range(nb_layers):
                        losses[i].zero_gradients()

                    for i in range(nb_layers):
                        losses[i].backprop()

                    # loss_result = loss(x_batch, y_batch)
                    # loss_results.append(loss_result)
                    # losses[-1].zero_gradients()
                    # losses[-1].backprop()

                    optimiser.step()

                # we compute the val losses:
                val_loss_results = []
                for loss in losses:
                    val_loss_result = loss(x_t, y_t)
                    val_loss_results.append(val_loss_result)
                val_loss_values = [
                    val_loss_result.nparray.mean()
                    for val_loss_result in val_loss_results
                ]

                if not best_val_loss_values:
                    best_val_loss_values = val_loss_values.copy()

                for i in range(nb_layers):
                    if val_loss_values[i] < best_val_loss_values[i]:
                        print(f"loss improvement for layer {i}, saving state")
                        best_val_loss_values[i] = val_loss_values[i]
                        best_state[i] = dense_layers[i].state

                print(f"current losses: ", end='')
                for val_loss_value in val_loss_values:
                    print(f"{val_loss_value}, ", end='')
                print("")
                print(f"best    losses: ", end='')
                for best_val_loss_value in best_val_loss_values:
                    print(f"{best_val_loss_value}, ", end='')
                print("")

                print(f"pass-through  : ", end='')
                for dense in dense_layers:
                    print(f"{dense.weights.nparray[0, 0]}, ", end='')
                print("")

                optimiser.learning_rate *= 0.98

    except KeyboardInterrupt:
        pass
    # Inference:
    for i in range(nb_layers):
        dense_layers[i].state = best_state[i]
    denoised = [
        module(tensor_class(x)).nparray.reshape(image.shape) for module in modules
    ]
    denoised = [numpy.clip(i, 0, 1) for i in denoised]
    ssims = [ssim(i, image) for i in denoised]
    for i in range(nb_layers):
        print(f"denoised{i}: {ssims[i]}")
    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        for i in denoised:
            viewer.add_image(n(i), name='denoised')
    assert ssims[-1] > 0.84


denoise_image(nb_layers=6, tensor_class=CLTensor)
