import numpy as np

from aydin.util.log.log import lsection, lprint


def unet_checks(input_dim, num_lyr, supervised, shiftconv):
    """
    This is to check if the input data is compatible with the model.
    :param input_dim: input dimension
    :param num_lyr: number of layers
    :param supervised: supervised or unsupervised
    :param shiftconv: whether it is shift convolution or masking architecture
    :return: assertion message
    """
    if None not in input_dim:
        with lsection(f'Pre-checking for CNN model generation:'):
            if supervised:
                shiftconv = False
                lprint(
                    'Shift convolution will be turned off automatically because supervised learning was selected.'
                )
            # assert (
            #     supervised != shiftconv
            # ), 'Shift convolution scheme is only available for self-supervised learning.'
            if shiftconv:
                assert (
                    np.mod(input_dim[:-1], np.repeat(2 ** num_lyr, len(input_dim[:-1])))
                    == 0
                ).all(), 'Each dimension of the input image has to be a multiple of 2^num_layer for shiftconv. '
            if supervised:
                lprint('Model will be created for supervised learning.')
            elif not supervised and shiftconv:
                lprint(
                    'Model will be generated for self-supervised learning with shift convlution scheme.'
                )
                assert (
                    np.diff(input_dim[:2]) == 0
                ), 'Make sure the input image shape is cubic as shiftconv mode involves rotation.'
                assert (
                    np.mod(
                        input_dim[:-1],
                        np.repeat(2 ** (num_lyr - 1), len(input_dim[:-1])),
                    )
                    == 0
                ).all(), (
                    'Each dimension of the input image has to be a multiple of '
                    '2^(num_layer-1) as shiftconv mode involvs pixel shift. '
                )
            elif not supervised and not shiftconv:
                lprint(
                    'Model will be generated for self-supervised with moving-blind spot scheme.'
                )

    return shiftconv
