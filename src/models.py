import typing as tp
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfrk
import tensorflow_recommenders as tfrs

from .constants import INTEGER, PLAIN, RELATIVE
from .layers import (NeighborEmbedding, PositionEmbedding,
                     RelativePositionEmbedding)
from .model_utils import (check_embedding_type, check_feature_type,
                          check_position_embedding_type)


class Node2Vec(tf.keras.models.Model):
    """
    Node2Vec
    --------
    """
    def __init__(self,
                embedding_dim: int,
                target_feature: str,
                context_feature: str,
                feature_type: str,
                feature_vocab: np.ndarray):
        check_feature_type(feature_type)
        super().__init__()
        self._target_feature = target_feature
        self._context_feature = context_feature
        feature_size = feature_vocab.shape[0] + 1

        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token=0)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token="_PAD_")

        self.embedding_layer = tf.keras.layers.Embedding(
            feature_size, 
            embedding_dim,
            mask_zero=True, 
            embeddings_initializer='he_normal',
            embeddings_regularizer= tf.keras.regularizers.l2(1e-6),
            name="embeddings",
            )
        self.dot = tf.keras.layers.Dot(axes=1, normalize=False)
    
    def call(self, inputs: tp.Dict[str, tf.Tensor]) -> tf.Tensor:
        target_lookup = self.lookup_layer(inputs[self._target_feature])
        context_lookup = self.lookup_layer(inputs[self._context_feature])
        target_embeddings = self.embedding_layer(target_lookup)
        context_embeddings = self.embedding_layer(context_lookup)
        logits = self.dot([target_embeddings, context_embeddings])
        return logits

class IRMAE(tf.keras.models.Model):
    """
    Implicit Rank-Minimizing Autoencoder
    ------------------------------------
    """
    def __init__(self, encoder: tf.keras.models.Model, decoder: tf.keras.models.Model, num_penalty_layers: int = 2):
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

    def __init__(self, feature_name: str, feature_type: str, feature_vocab: np.ndarray, embedding_dim: int, **embedding_kwargs):
        check_feature_type(feature_type)
        
        super().__init__()
        self.feature_name = feature_name
        feature_size = feature_vocab.shape[0] + 1
        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token=0)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token="_PAD_")
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_dim, mask_zero=True, **embedding_kwargs)
        
    def call(self, inputs: tp.Dict[str, tf.Tensor]):
        feature_lookup = self.lookup_layer(inputs[self.feature_name])
        return self.embedding_layer(feature_lookup) # output shape (batch_size, embedding_dim)

class LSTMEmbeddingModel(tf.keras.models.Model):
    """
    LSTM Embedding Model
    ---------------------
    """
    def __init__(self, feature_name: str, feature_vocab: tp.List[tp.Any], embedding_dim: int, num_recurrent_units: int, feature_type: str = "str"):
        check_feature_type(feature_type)
        
        super().__init__()
        self.feature_name = feature_name
        feature_size = feature_vocab.shape[0] + 1
        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token=0)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token="_PAD_")
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(num_recurrent_units)
        
    def call(self, inputs: tp.Dict[str, tf.Tensor]):
        feature_lookup = self.lookup_layer(inputs[self.feature_name])
        feature_embedding = self.embedding_layer(feature_lookup)
        return self.lstm(feature_embedding)

class AttentionDCN(tf.keras.models.Model):
    """
    Attention Deep Cross Network
    ----------------------------
    """
    def __init__(self, feature_name: str, feature_type: str, feature_vocab: np.ndarray, embedding_dim: int, deep_model_layer_sizes: int, use_cross_layer: bool, sequence_length: int, *, position_embeddings: str = None, embedding_type: str = 'plain', projection_dim: int = 32, conv_filters: int = 4, num_heads: int = 1, attn_dropout: float = 0.15, **embedding_kwargs): 
        check_feature_type(feature_type)
        check_embedding_type(embedding_type)

        super().__init__()
        self.feature_name = feature_name
        feature_size = feature_vocab.shape[0] + 1

        if feature_type == INTEGER:
            self.lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token=0)
        else:
            self.lookup_layer = tf.keras.layers.StringLookup(max_tokens=feature_size, vocabulary=feature_vocab, oov_token="_PAD_")

        self.query_layer_feature = tf.keras.layers.Conv1D(filters=embedding_dim, padding="same", kernel_size=conv_filters, input_shape=(sequence_length, embedding_dim))

        self.attention = tf.keras.layers.Attention(use_scale=True, causal=True, dropout=attn_dropout)
        self.average_pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.position_embedding = None
        if position_embeddings:
            check_position_embedding_type(position_embeddings)
            self.query_layer_position = tf.keras.layers.Conv1D(filters=embedding_dim, padding="same", kernel_size=conv_filters, input_shape=(sequence_length ,embedding_dim))
                        
            if position_embeddings == RELATIVE:
                self.position_embedding = RelativePositionEmbedding(output_dim=embedding_dim)
            else:
                self.position_embedding = PositionEmbedding(input_dim=feature_size, output_dim=embedding_dim)

        if embedding_type == PLAIN:
            self.embedding_layer = tf.keras.layers.Embedding(input_dim=feature_size, output_dim=embedding_dim, mask_zero=True)
        else:
            if "known_items_relations" not in embedding_kwargs:
                raise TypeError("`known_items_relations` parameter must be specified if using neigbor embeddings")
            known_items_relations = embedding_kwargs.pop("known_items_relations")
            self.embedding_layer = NeighborEmbedding(input_dim=feature_size, output_dim=embedding_dim, mask_zero=True, known_items_relations=known_items_relations, **embedding_kwargs)

        self.cross_layer = None
        if use_cross_layer:
            self.cross_layer = tfrs.layers.dcn.Cross(projection_dim, kernel_initializer="glorot_uniform")
        self.deep_model = [tf.keras.layers.Dense(num_units, activation='relu') for num_units in deep_model_layer_sizes]
        self.candidate_layer = tf.keras.layers.Dense(embedding_dim)
    
    def call(self, input: tp.Dict[str, tf.Tensor]):
        feature_lookups = self.lookup_layer(input[self.feature_name])
        item_embedding = self.embedding_layer(feature_lookups)

        item_query = self.query_layer_feature(item_embedding)

        if self.position_embedding is not None:
            position_embedding = self.position_embedding(input[self.feature_name])
            position_query = self.query_layer_position(position_embedding)
            q = tf.add(item_query, position_query)
            attn = self.attention([q, q])
        else:
            attn = self.attention([item_query, item_query])
        
        attn = self.average_pooling(attn)

        if self.cross_layer is not None:
            attn = self.cross_layer(attn)
        
        for layer in self.deep_model:
            attn = layer(attn)
        
        out = self.candidate_layer(attn)
        return out

class RetrievalModel(tfrs.Model):
    """
    Baseline for Retrieval Modelв 
    ----------------------------
    """
    def __init__(self,
                query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset
                ) -> None:
        super().__init__()
        self.query_model = query_model
        self.candidate_model = candidate_model

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
        query_embeddings = self.query_model(features)
        embedding_type = self.candidate_model(features)

        return self.task(
            query_embeddings, embedding_type, compute_metrics=not training)

# 1. Define Knowledge Distillation Retrieval Model
class KDRetrievalModel(tfrs.Model):
    """
    Retrieval Model with Knowledge Distillation capabilities
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                temperature: float = 3.0,
                alpha: float = 0.1) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
        # Get teacher embeddings (without gradient)
        teacher_embeddings = self.teacher_query_model(features)
        
        # Get student embeddings
        student_embeddings = self.student_query_model(features)
        
        # Get candidate embeddings
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # KD loss - make student embeddings similar to teacher
        # Apply temperature scaling for softening
        teacher_norm = tf.nn.l2_normalize(teacher_embeddings / self.temperature, axis=1)
        student_norm = tf.nn.l2_normalize(student_embeddings / self.temperature, axis=1)
        
        # Cosine similarity as a distillation loss
        kd_loss = -tf.reduce_mean(
            tf.reduce_sum(teacher_norm * student_norm, axis=1)
        ) * (self.temperature**2)
        
        # Combined loss
        return self.alpha * task_loss + (1 - self.alpha) * kd_loss


# 2. Feature-Based Distillation (FitNets) for Retrieval Model
class FitNetsRetrievalModel(tfrs.Model):
    """
    FitNets implementation for Retrieval Model
    Adds intermediate layer matching between teacher and student
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                hint_weight: float = 100.0,
                layer_to_take_from_student: str = None,
                layer_to_take_from_teacher: str = None) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # Hint weight for FitNets
        self.hint_weight = hint_weight

        self.layer_to_take_from_student = layer_to_take_from_student or 'all'
        self.layer_to_take_from_teacher = layer_to_take_from_teacher or 'all'
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
        # Extract teacher's intermediate representation
        # Access specific layers - adjust according to your model's structure
        if (layer_name := self.layer_to_take_from_teacher) == 'all':
            teacher_layer_output = self.teacher_query_model(features)
        else:
            teacher_layer_output = self.teacher_query_model.get_layer(layer_name)(features)
        
        # Student forward pass - get intermediate and final outputs
        if (layer_name := self.layer_to_take_from_student) == 'all':
            student_layer_output = self.student_query_model(features)
        else:
            student_layer_output = self.student_query_model.get_layer(layer_name)(features)
        student_embeddings = self.student_query_model(features)
         
        # Hint loss (FitNets approach)
        hint_loss = tf.reduce_mean(tf.square(student_layer_output - teacher_layer_output))
        
        # Get candidate embeddings
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # Combined loss
        return task_loss + self.hint_weight * hint_loss


# 3. Attention Transfer for Retrieval Model
class AttentionTransferRetrievalModel(tfrs.Model):
    """
    Attention Transfer implementation for Retrieval Model
    Transfers attention maps between teacher and student
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                attention_beta: float = 1000.0,
                layer_to_take_from_student: str = None,
                layer_to_take_from_teacher: str = None) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # Attention weight
        self.attention_beta = attention_beta
        
        self.layer_to_take_from_student = layer_to_take_from_student or 'all'
        self.layer_to_take_from_teacher = layer_to_take_from_teacher or 'all'
        

        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
    
    def attention_map(self, features):
        """Creates attention map by channel-wise sum of absolute values"""
        return tf.reduce_sum(tf.abs(features), axis=-1, keepdims=True)
    
    def compute_loss(self, features, training=False):
        # Get teacher's category history and attention map
        if (layer_name := self.layer_to_take_from_teacher) == 'all':
            teacher_layer_output = self.teacher_query_model(features)
        else:
            teacher_layer_output = self.teacher_query_model.get_layer(layer_name)(features)

        teacher_attention = self.attention_map(teacher_layer_output)
        
        # Get student's category history and attention map
        student_category = self.student_query_model.category_history_1(features)
        student_attention = self.attention_map(student_category)
        student_embeddings = self.student_query_model(features)
        
        # Attention transfer loss
        attention_loss = tf.reduce_mean(tf.square(student_attention - teacher_attention))
        
        # Get candidate embeddings
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # Combined loss
        return task_loss + self.attention_beta * attention_loss


# 4. Relational Knowledge Distillation for Retrieval Model
class RKDRetrievalModel(tfrs.Model):
    """
    Relational Knowledge Distillation implementation for Retrieval Model
    Preserves structural relationships between samples
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                distance_weight: float = 25.0) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # RKD weight
        self.distance_weight = distance_weight
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
    
    def pairwise_distance(self, embeddings):
        """Compute pairwise distances between embeddings"""
        square = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
        distances = square - 2 * tf.matmul(embeddings, tf.transpose(embeddings)) + tf.transpose(square)
        distances = tf.maximum(distances, 0.0)
        # Zero out diagonal
        mask = tf.ones_like(distances) - tf.eye(tf.shape(distances)[0])
        distances = distances * mask
        return distances
    
    def compute_loss(self, features, training=False):
        # Get teacher embeddings
        teacher_embeddings = self.teacher_query_model(features)
        
        # Get student embeddings
        student_embeddings = self.student_query_model(features)
        
        # Compute pairwise distances
        t_dist = self.pairwise_distance(teacher_embeddings)
        s_dist = self.pairwise_distance(student_embeddings)
        
        # Normalize distances
        mean_td = tf.reduce_mean(t_dist) + 1e-6
        mean_sd = tf.reduce_mean(s_dist) + 1e-6
        t_dist = t_dist / mean_td
        s_dist = s_dist / mean_sd
        
        # RKD distance loss
        rkd_loss = tf.reduce_mean(tf.square(t_dist - s_dist))
        
        # Get candidate embeddings
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # Combined loss
        return task_loss + self.distance_weight * rkd_loss


# 5. Correlation Congruence for Retrieval Model
class CorrelationCongruenceRetrievalModel(tfrs.Model):
    """
    Correlation Congruence implementation for Retrieval Model
    Transfers correlation structure between feature maps
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                cc_weight: float = 10.0,
                layer_to_take_from_student: str = None,
                layer_to_take_from_teacher: str = None) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # Correlation Congruence weight
        self.cc_weight = cc_weight

        self.layer_to_take_from_student = layer_to_take_from_student or 'all'
        self.layer_to_take_from_teacher = layer_to_take_from_teacher or 'all'
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
    
    def correlation_matrix(self, features):
        """Compute correlation matrix of features"""
        features = tf.reshape(features, [tf.shape(features)[0], -1])
        features = features - tf.reduce_mean(features, axis=0, keepdims=True)
        features_norm = tf.norm(features, axis=0, keepdims=True)
        features_normalized = features / (features_norm + 1e-10)
        correlation = tf.matmul(tf.transpose(features_normalized), features_normalized)
        return correlation
    
    def compute_loss(self, features, training=False):

        if (layer_name := self.layer_to_take_from_teacher) == 'all':
            teacher_layer_output = self.teacher_query_model(features)
        else:
            teacher_layer_output = self.teacher_query_model.get_layer(layer_name)(features)

        if (layer_name := self.layer_to_take_from_student) == 'all':
            student_layer_output = self.student_query_model(features)
        else:
            student_layer_output = self.student_query_model.get_layer(layer_name)(features)

        student_embeddings = self.student_query_model(features)
        
        # Compute correlation matrices
        teacher_corr = self.correlation_matrix(teacher_layer_output)
        student_corr = self.correlation_matrix(student_layer_output)
        
        # Correlation congruence loss
        cc_loss = tf.reduce_mean(tf.square(teacher_corr - student_corr))
        
        # Get candidate embeddings
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # Combined loss
        return task_loss + self.cc_weight * cc_loss


# 6. Contrastive Representation Distillation for Retrieval Model
class CRDRetrievalModel(tfrs.Model):
    """
    Contrastive Representation Distillation for Retrieval Model
    Uses contrastive learning to transfer representations
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                temperature: float = 0.1,
                contrastive_weight: float = 10.0) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # CRD parameters
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
        # Get teacher and student embeddings
        teacher_embeddings = self.teacher_query_model(features)
        student_embeddings = self.student_query_model(features)
        
        # Normalize embeddings
        teacher_norm = tf.nn.l2_normalize(teacher_embeddings, axis=1)
        student_norm = tf.nn.l2_normalize(student_embeddings, axis=1)
        
        # Compute similarity matrix
        similarity = tf.matmul(student_norm, teacher_norm, transpose_b=True) / self.temperature
        
        # Create labels (diagonal is positive pairs)
        batch_size = tf.shape(student_embeddings)[0]
        labels_crd = tf.eye(batch_size)
        
        # Contrastive loss (InfoNCE style)
        crd_loss = tf.keras.losses.categorical_crossentropy(
            labels_crd, tf.nn.softmax(similarity), from_logits=False)
        
        # Get candidate embeddings
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # Combined loss
        return task_loss + self.contrastive_weight * crd_loss


# 7. Progressive Layer-wise Distillation for Retrieval Model
class ProgressiveLayerwiseDistillationRetrievalModel(tfrs.Model):
    """
    Progressive Layer-wise Distillation for Retrieval Model
    Train the model layer by layer
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                current_stage: int = 0,
                hint_weight: float = 10.0,
                layer_to_take_from_student: str = None,
                layer_to_take_from_teacher: str = None) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # Training stage (0: train AttentionDCN only, 1: train full model)
        self.current_stage = current_stage
        
        # Hint weight
        self.hint_weight = hint_weight

        self.layer_to_take_from_student = layer_to_take_from_student or 'all'
        self.layer_to_take_from_teacher = layer_to_take_from_teacher or 'all'
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
        
        # Set trainable variables based on stage
        self.set_trainable_vars()
    
    def set_trainable_vars(self):
        """Set which parts of the student model are trainable based on current stage"""
        # Reset all variables to trainable
        for layer in self.student_query_model.layers:
            layer.trainable = True
            
        if self.current_stage == 0:
            # Stage 1: Train only the AttentionDCN part
            # Freeze other parts of the model
            for layer in self.student_query_model.layers:
                if 'category_history_1' not in layer.name:
                    layer.trainable = False
        # For stage 1, everything remains trainable
    
    def advance_stage(self):
        """Move to the next training stage"""
        if self.current_stage < 1:
            self.current_stage += 1
            self.set_trainable_vars()
            return True
        return False
    
    def compute_loss(self, features, training=False):
        if self.current_stage == 0:
            # Stage 1: Match intermediate representations
            if (layer_name := self.layer_to_take_from_teacher) == 'all':
                teacher_layer_output = self.teacher_query_model(features)
            else:
                teacher_layer_output = self.teacher_query_model.get_layer(layer_name)(features)

            if (layer_name := self.layer_to_take_from_student) == 'all':
                student_layer_output = self.student_query_model(features)
            else:
                student_layer_output = self.student_query_model.get_layer(layer_name)(features)

            # Hint loss for AttentionDCN output
            hint_loss = tf.reduce_mean(tf.square(student_layer_output - teacher_layer_output))
            
            # No task loss in first stage
            return hint_loss
        else:
            # Stage 2: Match final embeddings while freezing early layers
            teacher_embeddings = self.teacher_query_model(features)
            student_embeddings = self.student_query_model(features)
            
            # Distillation loss for final embeddings
            hint_loss = tf.reduce_mean(tf.square(student_embeddings - teacher_embeddings))
            
            # Get candidate embeddings
            candidate_embeddings = self.candidate_model(features)
            
            # Original retrieval task loss
            task_loss = self.task(
                student_embeddings, candidate_embeddings, compute_metrics=not training)
            
            return task_loss + self.hint_weight * hint_loss


# 8. Born-Again Networks for Retrieval Model
class BornAgainRetrievalModel(tfrs.Model):
    """
    Born Again Networks for Retrieval Model
    Implements the Born-Again Networks approach (Furlanello et al., 2018) for recommender systems
    Iteratively trains a sequence of student networks, each one learning from the previous generation
    """
    def __init__(self,
                teacher_query_model: tf.keras.models.Model,
                student_query_model: tf.keras.models.Model,
                candidate_model: tf.keras.models.Model,
                candidate_pool: tf.data.Dataset,
                temperature: float = 3.0,
                alpha: float = 0.1,
                generation: int = 0) -> None:
        super().__init__()
        
        # Teacher model (frozen)
        self.teacher_query_model = teacher_query_model
        self.teacher_query_model.trainable = False
        
        # Student model (to be trained)
        self.student_query_model = student_query_model
        
        # Candidate model
        self.candidate_model = candidate_model
        
        # Reference to candidate pool for creating next generation models
        self.candidate_pool = candidate_pool
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha
        
        # Track which generation this model represents
        self.generation = generation
        
        # Retrieval task for student
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_pool.batch(1024).map(self.candidate_model),
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
        """Compute the loss combining retrieval task loss and knowledge distillation loss"""
        # Get teacher embeddings (without gradient)
        teacher_embeddings = self.teacher_query_model(features)
        
        # Get student embeddings
        student_embeddings = self.student_query_model(features)
        
        # Get candidate embeddings for retrieval task
        candidate_embeddings = self.candidate_model(features)
        
        # Original retrieval task loss
        task_loss = self.task(
            student_embeddings, candidate_embeddings, compute_metrics=not training)
        
        # Apply temperature scaling for KD
        teacher_norm = tf.nn.l2_normalize(teacher_embeddings / self.temperature, axis=1)
        student_norm = tf.nn.l2_normalize(student_embeddings / self.temperature, axis=1)
        
        # KL divergence loss
        kd_loss = -tf.reduce_mean(
            tf.reduce_sum(teacher_norm * tf.math.log(student_norm + 1e-8), axis=1)
        ) * (self.temperature**2)
        
        # Combined loss
        return self.alpha * task_loss + (1 - self.alpha) * kd_loss
    

def create_next_generation_born_again_model(prev_gen: BornAgainRetrievalModel,
                                             student_model: tp.Callable[..., tf.keras.models.Model]):
    """Create the next generation model using this model's student as the new teacher"""
    # Create new student with same architecture as current student
    new_student_model = student_model(
        deep_model_layer_sizes=[64],
        embedding_dim=prev_gen.student_query_model.dense_layer.units
    )
    
    # Create a new model with current student as teacher
    next_gen = BornAgainRetrievalModel(
        teacher_query_model=prev_gen.student_query_model,
        student_query_model=new_student_model,
        candidate_model=prev_gen.candidate_model,
        candidate_pool=prev_gen.candidate_pool,
        temperature=prev_gen.temperature,
        alpha=prev_gen.alpha,
        generation=prev_gen.generation + 1
    )
    
    return next_gen
