import math
import typing as tp
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.initializers import Initializer
from tensorflow.python.ops import embedding_ops, math_ops


class NeighborEmbedding(tf.keras.layers.Embedding):
    """
    Neighbor Embedding
    ------------------
    Arguments:
    ----------
        input_dim: Size of sequence.
        output_dim: Size of the hidden layer.
    """
    def __init__(self, input_dim: int, output_dim: int, *, known_items_relations: tf.SparseTensor, embeddings_initializer: tp.Union[str, Initializer] = 'uniform', **kwargs):
        if 'decay_rate' in kwargs:
            self.decay_rate = kwargs.pop('decay_rate')
        self.decay_rate = 0.5
        self._A = known_items_relations
        super().__init__(input_dim, output_dim, embeddings_initializer=embeddings_initializer, **kwargs)

    def get_config(self) -> tp.Dict[str, tp.Any]:
        config = {
            "decay_rate": self.decay_rate,
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _select_neighbors_for_index(self, token: int):
        neighbor_index = tf.squeeze(
                    tf.gather(self._A.indices, tf.where(self._A.indices[:,0] == token))[:,:,1], # second index is neighbor index
                    axis=[-1])
        neighbor_embeddings = embedding_ops.embedding_lookup_v2(self.embeddings, neighbor_index)
        return tf.reduce_sum(neighbor_embeddings, axis=0)
    
    def _get_aggregated_neighbor_embeddings(self, token: tf.Tensor) -> tf.Tensor:
        neighbors = tf.map_fn(self._select_neighbors_for_index, token, fn_output_signature=tf.float32)
        return neighbors
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Arguments:
        ----------
            inputs: A tensor tuple of sequences `inputs` of size `(batch_size, sequence_length)` 
            and adjacency matrix `A` of size `(known_vocab_size, known_vocab_size)`
        Returns:
        --------
            A tensor in shape of `(length, output_dim)`.
        """
        dtype = backend.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        out = embedding_ops.embedding_lookup_v2(self.embeddings, inputs)
        out = tf.multiply(out, 1 - self.decay_rate)
        out_neighbors = tf.map_fn(self._get_aggregated_neighbor_embeddings, inputs, fn_output_signature=tf.float32)
        out_neighbors = tf.multiply(out_neighbors, self.decay_rate)
        out = out + out_neighbors
        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            out = math_ops.cast(out, self._dtype_policy.compute_dtype)
        return out

class PositionEmbedding(tf.keras.layers.Embedding):
    """
    Position Embedding
    ------------------
    Arguments:
    ----------
        input_dim: Size of sequence (maximm length).
        output_dim: Size of the hidden layer.
    """
    def __init__(self, input_dim: int, output_dim: int, *, embeddings_initializer: tp.Union[str, Initializer] = 'uniform', **kwargs):
        super().__init__(input_dim, output_dim, embeddings_initializer, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Arguments:
        ----------
            inputs: A tensor of size `(batch_size, sequence_length)` 
        Returns:
        --------
            A tensor in shape of `(length, output_dim)`.
        """
        input_shape = tf.shape(inputs[..., tf.newaxis])
        actual_seq_len = input_shape[1]
        position_embeddings = self.embeddings[tf.newaxis, :actual_seq_len, :]
        new_shape = tf.where([True, True, False], input_shape, tf.shape(position_embeddings))
        return tf.broadcast_to(position_embeddings, new_shape)

class RelativePositionEmbedding(tf.keras.layers.Layer):
    """
    Relative Position Embedding
    ---------------------------
    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized in
    "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).
    Arguments:
    ----------
        output_dim: Size of the hidden layer.
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position.
    """
    def __init__(self,
               output_dim: int,
               *,
               min_timescale: float = 1.0,
               max_timescale: float = 1.0e4,
               **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def get_config(self) -> tp.Dict[str, tp.Any]:
        config = {
            "output_dim": self.output_dim,
            "min_timescale": self.min_timescale,
            "max_timescale": self.max_timescale,
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, length: int = None) -> tf.Tensor:
        """
        Arguments:
        ----------
            inputs: A tensor whose second dimension will be used as `length`. If
                `None`, the other `length` argument must be specified.
            length: An optional integer specifying the number of positions. If both
                `inputs` and `length` are spcified, `length` must be equal to the second
                dimension of `inputs`.
        Returns:
        --------
            A tensor in shape of `(length, output_dim)`.
        """
        if inputs is None and length is None:
            raise ValueError("If inputs is None, `length` must be set in "
                        "RelativePositionEmbedding().")
        if inputs is not None:
            input_shape = backend.shape(inputs) # (batch_size, sequence_length)
            if length is not None and length != input_shape[1]:
                    raise ValueError(
                        "If inputs is not None, `length` must equal to input_shape[1].")
            length = input_shape[1]
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._output_dim // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) *
            -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        position_embeddings = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return position_embeddings
