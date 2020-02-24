import tensorflow as tf

K = tf.keras.backend
Lambda = tf.keras.layers.Lambda


# The most simple and working implementation.
def split(x, idx, batchsize=1, lyrname=None):  #
    """
    Split tensor at the batch axis. Only for shift convolution architecture.
    :param x: input tensor
    :param idx: index for the split chunk
    :param batchsize: batch size
    :param lyrname: layer name
    """
    out_shape = K.int_shape(x[0])
    return Lambda(
        lambda xx: xx[idx * batchsize : (idx + 1) * batchsize],
        output_shape=out_shape,
        name=lyrname,
    )
