from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers, constraints, initializers
import tensorflow.keras.backend as K

dot = tf.matmul


class SpectralNorm(tf.keras.constraints.Constraint):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter

    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        for _ in range(self.n_iter):
            v = dot(w, u, transpose_a=True)
            v /= tf.norm(v)
            u = dot(w, v)
            u /= tf.norm(u)
        spec_norm = dot(u, tf.matmul(w, v), transpose_a=True)
        return input_weights / spec_norm


class Attention(keras.layers.Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 trainable=True)
        self.features_dim = input_shape[-1]
        self.step_dim = input_shape[-2]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     trainable=True)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class GCNConv(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 p,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        self.units = units
        self.P = p
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(GCNConv, self).__init__()

    def build(self, input_shape):
        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(self.units, self.units),
                                          initializer=self.kernel_initializer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        super(GCNConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        Ans = inputs[0]
        X = inputs[1]
        h = dot(X, self.weight)
        h = tf.transpose(h, [0, 2, 1])
        output = []
        for An, h1 in zip(tf.unstack(Ans, axis=1), tf.split(h, axis=1, num_or_size_splits=self.P)):
            degree = tf.reduce_sum(An, axis=1)
            degree_l = tf.linalg.diag(degree)
            diagonal_degree_hat = tf.linalg.diag(1 / (tf.sqrt(degree) + 1e-7))
            laplacain = dot(diagonal_degree_hat, dot(degree_l - An, diagonal_degree_hat))
            output.append(dot(h1, laplacain))

        output = tf.transpose(tf.concat(output, axis=1), [0, 2, 1])

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output
