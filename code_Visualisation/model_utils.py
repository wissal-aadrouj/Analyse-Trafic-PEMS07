import tensorflow as tf
from tensorflow.keras.layers import Layer

class GCNLayer(Layer):
    def __init__(self, adj, units, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        # On s'assure que l'adjacence est stockée correctement
        self.adj = tf.constant(adj, dtype=tf.float32)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Propagation spatiale (A * X) puis multiplication par W
        x = tf.einsum('ij,btjf->btif', self.adj, inputs)
        x = tf.einsum('btif,fc->btic', x, self.W)
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "adj": self.adj.numpy().tolist()
        })
        return config