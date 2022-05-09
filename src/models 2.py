from collections import defaultdict
import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfrk

from layers import (NeighborEmbedding, 
                    PositionEmbedding, 
                    RelativePositionEmbedding)
from model_utils import check_feature_type, check_position_embedding_type, check_embedding_type
from constants import INTEGER, RELATIVE, PLAIN

class Node2Vec(tf.keras.models.Model):
    """
    Node2Vec
    --------
    """
    def __init__(self,
                input_dim: int, 
                embedding_dim: int,
                target_feature: str,
                context_feature: str):
        super().__init__()
        self._target_feature = target_feature
        self._context_feature = context_feature

        self.lookup_layer = tf.keras.layers.IntegerLookup()

        self.embedding_layer_layer = tf.keras.layers.Embedding(
            input_dim, 
            embedding_dim, 
            embeddings_initializer='he_normal',
            embeddings_regularizer= tf.keras.regularizers.l2(1e-6),
            name="embedding_type",
            )
        self.dot = tf.keras.layers.Dot(axes=1, normalize=False)
    
    def call(self, inputs: tp.Dict[str, tf.Tensor]) -> tf.Tensor:
        target_embeddings = self.embedding_layer_layer(self._target_feature)
        context_embeddings = self.embedding_layer_layer(self._context_feature)
        logits = self.dot(target_embeddings, context_embeddings)
        return logits

class IRMAE(tf.keras.models.Model):
    """
    Implicit Rank-Minimizing Autoencoder
    ------------------------------------
    """
    def __init__(self, 
                encoder: tf.keras.models.Model, 
                decoder: tf.keras.models.Model, 
                num_penalty_layers: int = 2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        encoder_output = self.encoder.layers[-1].output_shape[-1]
        self.penalty_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=encoder_output)
            for _ in range(num_penalty_layers)
        ])

    def call(self, inputs) -> tf.Tensor:
        encoder_output = self.encoder(inputs)
        encoder_output = self.penalty_model(encoder_output)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

class PlainEmbeddingModel(tf.keras.models.Model):
    """
    Basic Embedding Model
    ---------------------
    """

    def __init__(self, feature_name: str, feature_type: str, feature_vocab: np.ndarray, embedding_dim: int, **kwargs):
        check_feature_type(feature_type)
        
        super().__init__()
        self.feature_name = feature_name
        feature_size = feature_vocab.shape[0]
        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab)
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_dim)
        
    def call(self, inputs: tp.Dict(tf.Tensor)):
        feature_lookup = self.lookup_layer(inputs[self.feature_name])
        return self.embedding_layer(feature_lookup)

class LSTMEmbeddingModel(tf.keras.models.Model):
    """
    LSTM Embedding Model
    ---------------------
    """
    def __init__(self, feature_name: str, feature_vocab: tp.List[tp.Any], embedding_dim: int, num_recurrent_units: int, feature_type: str = "str"):
        check_feature_type(feature_type)
        
        super().__init__()
        self.feature_name = feature_name
        feature_size = feature_vocab.shape[0]
        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab)
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(num_recurrent_units)
        
    def call(self, inputs: tp.Dict(tf.Tensor)):
        feature_lookup = self.lookup_layer(inputs[self.feature_name])
        feature_embedding = self.embedding_layer(feature_lookup)
        return self.lstm(feature_embedding)

class AttentionDCN(tf.keras.models.Model):
    """
    Attention Deep Cross Network
    ----------------------------
    """
    def __init__(self,
                feature_name: str,
                feature_type: str,
                feature_vocab: np.ndarray,
                embedding_dim: int,
                deep_model_layer_sizes: int,
                use_cross_layer: bool,
                *,
                position_embeddings: str = None,
                embedding_type: str = 'plain',
                projection_dim: int = 32,
                **embedding_kwargs): 
        check_feature_type(feature_type)
        check_embedding_type(embedding_type)
        check_position_embedding_type(position_embeddings)

        super().__init__()
        self.feature_name = feature_name
        feature_size = feature_vocab.shape[0]


        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab)

        self.query_layer_feature = tf.keras.layers.Dense(units=feature_size)
        self.key_layer_feature = tf.keras.layers.Dense(units=feature_size)
        self.value_layer_feature = tf.keras.layers.Dense(units=feature_size)

        self.query_layer_position = tf.keras.layers.Dense(units=feature_size)
        self.key_layer_position = tf.keras.layers.Dense(units=feature_size)
        self.value_layer_position = tf.keras.layers.Dense(units=feature_size)

        self.attention = tf.keras.layers.Attention(use_scale=True, causal=True)
        self.average_pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.position_embedding = None
        if position_embeddings == RELATIVE:
            self.position_embedding = RelativePositionEmbedding(output_dim=embedding_dim)
        else:
            self.position_embedding = PositionEmbedding(input_dim=feature_size, output_dim=embedding_dim)

        if embedding_type == PLAIN:
            self.embedding_layer = tf.keras.layers.Embedding(input_dim=feature_size, output_dim=embedding_dim, mask_zero=True)
        else:
            if "known_items_relations" not in embedding_kwargs:
                raise TypeError("`known_items_relations` parameter must be specified if using neigbor embeddings")
            known_feature_relations = embedding_kwargs.pop("known_items_relations")
            self.embedding_layer = NeighborEmbedding(input_dim=feature_size, output_dim=embedding_dim, mask_zero=True, known_feature_relations=known_feature_relations, **embedding_kwargs)

        self.cross_layer = None
        if use_cross_layer:
            self.cross_layer = tfrs.layers.dcn.Cross(projection_dim, kernel_initializer="glorot_uniform")
        self.deep_model = [tf.keras.layers.Dense(num_units, activation='relu') for num_units in deep_model_layer_sizes]
        self.candidate_layer = tf.keras.layers.Dense(embedding_dim)
    
    def call(self, input: tp.Dict[str, tf.Tensor]):
        feature_lookups = self.lookup_layer(input[self.feature_name])
        item_embedding = self.embedding_layer(feature_lookups)

        item_query = self.query_layer_feature(item_embedding)
        item_key = self.key_layer_feature(item_embedding)
        item_value = self.value_layer_feature(item_embedding)

        if self.position_embedding is not None:
            position_embedding = self.position_embedding(input[self.feature_names])
            position_query = self.query_layer_position(position_embedding)
            position_key = self.key_layer_position(position_embedding)
            position_value = self.value_layer_position(position_embedding)

            q = tf.add(item_query, position_query)
            k = tf.add(item_key, position_key)
            v = tf.add(item_value, position_value)

            attn = self.attention([q, k, v])
        else:
            attn = self.attention([item_query, item_key, item_value])
        
        attn = self.average_pooling(attn)

        if self.cross_layer is not None:
            x_0 = self.cross_layer(attn)
        
        for layer in self.deep_model:
            x_0 = layer(x_0)
        
        out = self.candidate_layer(x_0)
        return out

class RetrievalModel(tfrs.Model):
    """
    Baseline for Retrieval Modelв 
    ----------------------------
    """
    def __init__(self,
                query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                query_features: tp.List[str],
                candidate_feature: str,
                ) -> None:
        super().__init__()
        self.query_model = query_model
        self.candidate_model = candidate_model
        self._query_features = query_features
        self._candidate_feature = candidate_feature

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(128).map(self.candidate_model),
                k=100,
                metrics=[
                        tfrk.keras.metrics.MeanAveragePrecisionMetric(topn=1, name='map_1'),
                        tfrk.keras.metrics.MeanAveragePrecisionMetric(topn=10, name='map_10'),
                        tfrk.keras.metrics.MeanAveragePrecisionMetric(topn=100, name='map_100'),
                        tfrk.keras.metrics.NDCGMetric(topn=1, name='ndcg_1'),
                        tfrk.keras.metrics.NDCGMetric(topn=10, name='ndcg_10'),
                        tfrk.keras.metrics.NDCGMetric(topn=100, name='ndcg_100'),
                        ]
            ),
        )

    def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        query_embeddings = self.query_model({feat: features[feat]
            for feat in self._query_features
        })
        embedding_type = self.candidate_model(features[self._candidate_feature])

        return self.task(
            query_embeddings, embedding_type, compute_metrics=not training)
