from tensorflow_core.python.keras.engine.base_layer import Layer
from tensorflow_core.python.keras.layers.merge import multiply


class Maskout(Layer):
    def __init__(self, **kwargs):
        """
        A layer that mutiply mask with image. This is for masking architecture.
        """
        super(Maskout, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(Maskout, self).build(input_shape)

    def call(self, x, training=None):
        assert isinstance(x, list)
        return multiply(x)
        # return K.in_train_phase(multiply(x), x[0], training=training)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a
