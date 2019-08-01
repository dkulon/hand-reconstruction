import tensorflow as tf

class MeshReLU1B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[1, n_channels]
        )

    def call(self, x):
        return tf.nn.relu(x + self.bias)


class MeshReLU2B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        _, n_vertexes, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[n_vertexes, n_channels])

    def call(self, x):
        return tf.nn.relu(x + self.bias)

class MeshLeakyReLU1B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[1, n_channels]
        )

    def call(self, x):
        return tf.nn.leaky_relu(x + self.bias)

class MeshLeakyReLU2B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        _, n_vertexes, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[n_vertexes, n_channels])

    def call(self, x):
        return tf.nn.leaky_relu(x + self.bias)

class MeshELU1B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[1, n_channels]
        )

    def call(self, x):
        return tf.nn.elu(x + self.bias)