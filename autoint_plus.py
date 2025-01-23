import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten, Activation, Input, Embedding, BatchNormalization
from tensorflow.keras.initializers import glorot_normal, Zeros, TruncatedNormal
from tensorflow.keras.regularizers import l2


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy


from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import math

class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.embedding = tf.keras.layers.Embedding(
                sum(field_dims), 
                embed_dim, 
                embeddings_initializer='glorot_uniform'  # TensorFlow's equivalent of xavier_uniform
        )
    
    def call(self, x):
        x = x + tf.constant(self.offsets, dtype=x.dtype)
        return self.embedding(x)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([tf.keras.initializers.GlorotUniform()(shape=self.embedding.weights[0].shape)])

class MultiLayerPerceptron(Layer):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, 
                 dropout_rate=0, use_bn=False, init_std=0.0001, output_layer=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        
        hidden_units = [inputs_dim] + list(hidden_units)
        if output_layer:
            hidden_units += [1]
        
        self.layers = []
        for i in range(len(hidden_units) - 1):
            # Linear layer
            layer = Dense(hidden_units[i+1], 
                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=init_std),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            self.layers.append(layer)
            
            # Batch Normalization
            if use_bn:
                self.layers.append(tf.keras.layers.BatchNormalization())
            
            # Activation
            self.layers.append(tf.keras.layers.Activation('relu'))
            
            # Dropout
            self.layers.append(tf.keras.layers.Dropout(dropout_rate))
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x

class MultiHeadSelfAttention(Layer):
    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False):
        super().__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        
        self.W_Query = tf.Variable(tf.random.normal((embedding_size, embedding_size), stddev=0.05))
        self.W_Key = tf.Variable(tf.random.normal((embedding_size, embedding_size), stddev=0.05))
        self.W_Value = tf.Variable(tf.random.normal((embedding_size, embedding_size), stddev=0.05))
        
        if self.use_res:
            self.W_Res = tf.Variable(tf.random.normal((embedding_size, embedding_size), stddev=0.05))
    
    def call(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(inputs.shape)}, expect to be 3 dimensions")
        
        # Linear transformations
        querys = tf.tensordot(inputs, self.W_Query, axes=1)
        keys = tf.tensordot(inputs, self.W_Key, axes=1)
        values = tf.tensordot(inputs, self.W_Value, axes=1)
        
        # Split heads
        querys = tf.stack(tf.split(querys, self.head_num, axis=-1))
        keys = tf.stack(tf.split(keys, self.head_num, axis=-1))
        values = tf.stack(tf.split(values, self.head_num, axis=-1))
        
        # Attention
        inner_product = tf.einsum('bnik,bnjk->bnij', querys, keys)
        if self.scaling:
            inner_product /= tf.sqrt(float(self.att_embedding_size))
        
        normalized_att_scores = tf.nn.softmax(inner_product, axis=-1)
        result = tf.matmul(normalized_att_scores, values)
        
        # Combine heads
        result = tf.concat(tf.unstack(result), axis=-1)
        
        # Residual connection
        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=1)
        
        return tf.nn.relu(result)

    def compute_output_shape(self, input_shape):

        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num
                  , 'use_res': self.use_res, 'seed': self.seed}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        base_config.update(config)
        return base_config


class AutoIntMLP(Layer):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, 
                 att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, 
                 dnn_dropout=0.4, init_std=0.0001):
        super().__init__()
        
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size
        self.att_output_dim = self.num_fields * self.embedding_size
        self.embed_output_dim = len(field_dims) * embedding_size
        
        self.dnn_linear = Dense(1, use_bias=False, 
                                kernel_initializer=tf.random_normal_initializer(stddev=init_std))
        
        self.dnn = MultiLayerPerceptron(
            self.embed_output_dim, 
            dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std
        )
        
        self.int_layers = [
            MultiHeadSelfAttention(
                self.embedding_size, 
                head_num=att_head_num, 
                use_res=att_res
            ) for _ in range(att_layer_num)
        ]
    
    def call(self, X, training=False):
        embed_x = self.embedding(X)
        dnn_embed = embed_x
        att_input = embed_x
        
        for layer in self.int_layers:
            att_input = layer(att_input)
        
        att_output = tf.reshape(att_input, [-1, self.att_output_dim])
        att_output = tf.nn.relu(self.dnn_linear(att_output))
        
        dnn_output = self.dnn(tf.reshape(dnn_embed, [-1, self.embed_output_dim]), training=training)
        
        y_pred = tf.sigmoid(att_output + dnn_output)
        return y_pred


class AutoIntMLPModel(tf.keras.Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, 
                 att_res=True, l2_reg_dnn=0, l2_reg_embedding=1e-5, 
                 dnn_hidden_units=(32, 32), dnn_activation='relu',
                 dnn_use_bn=False, dnn_dropout=0, init_std=0.0001):
        super().__init__()
        self.autoInt_mlp_layer = AutoIntMLP(
            field_dims, 
            embedding_size, 
            att_layer_num=att_layer_num, 
            att_head_num=att_head_num,
            att_res=att_res, 
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            l2_reg_dnn=l2_reg_dnn, 
            l2_reg_embedding=l2_reg_embedding,
            dnn_use_bn=dnn_use_bn, 
            dnn_dropout=dnn_dropout, 
            init_std=init_std
        )
    
    def call(self, inputs, training=False):
        return self.autoInt_mlp_layer(inputs, training=training)
    
    
def predict_model(model, pred_df):
    batch_size = 2048
    top=10
    user_pred_info = []
    total_rows = len(pred_df)
    for i in range(0, total_rows, batch_size):
        features = pred_df.iloc[i:i + batch_size, :].values
        y_pred = model.predict(features, verbose=False)
        for feature, p in zip(features, y_pred):
            u_i = feature[:2]
            user_pred_info.append((int(u_i[1]), float(p)))
    
    return sorted(user_pred_info, key=lambda s : s[1], reverse=True)[:top]