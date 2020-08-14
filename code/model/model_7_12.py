# -*- encoding: utf-8 -*-
'''
File    :   model.py
Time    :   2020/06/23 14:50:29
Author  :   Chao Wang 
Version :   1.0
Contact :   374494067@qq.com
@Desc    :   None
'''

'''
utils
'''
import math
from typing import Union, Callable, Optional
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils,get_custom_objects
from keras import initializers
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects
from keras import regularizers



'''
attention:
'''

class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
            
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]




'''
Optimizer
'''
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = self.total_steps - warmup_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr * (1.0 - K.minimum(t, decay_steps) / decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t > 5, r_t * m_corr_t / v_corr_t, m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).

    """
    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R
                
                model.train_function = F

################################################################

'''
utils
'''
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28



# class Attention(Layer):
#     def __init__(self, step_dim,
#                  W_regularizer=None, b_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#         """
#         Keras Layer that implements an Attention mechanism for temporal data.
#         Supports Masking.
#         Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
#         # Input shape
#             3D tensor with shape: `(samples, steps, features)`.
#         # Output shape
#             2D tensor with shape: `(samples, features)`.
#         :param kwargs:
#         Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
#         The dimensions are inferred based on the output shape of the RNN.
#         Example:
#             model.add(LSTM(64, return_sequences=True))
#             model.add(Attention())
#         """
#         self.supports_masking = True
#         #self.init = initializations.get('glorot_uniform')
#         self.init = initializers.get('glorot_uniform')

#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)

#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)

#         self.bias = bias
#         self.step_dim = step_dim
#         self.features_dim = 0
#         super(Attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         assert len(input_shape) == 3

#         self.W = self.add_weight(shape = (input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         self.features_dim = input_shape[-1]

#         if self.bias:
#             self.b = self.add_weight(shape = (input_shape[1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
#         else:
#             self.b = None

#         self.built = True

#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None

#     def call(self, x, mask=None):
#         # eij = K.dot(x, self.W) TF backend doesn't support it

#         # features_dim = self.W.shape[0]
#         # step_dim = x._keras_shape[1]

#         features_dim = self.features_dim
#         step_dim = self.step_dim

#         eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

#         if self.bias:
#             eij += self.b

#         eij = K.tanh(eij)

#         a = K.exp(eij)

#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())

#         # in some cases especially in the early stages of training the sum may be almost zero
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

#         a = K.expand_dims(a)
#         weighted_input = x * a
#     #print weigthted_input.shape
#         return K.sum(weighted_input, axis=1)

#     def compute_output_shape(self, input_shape):
#         #return input_shape[0], input_shape[-1]
#         return input_shape[0],  self.features_dim


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)

################################################################



'''
Transformer
https://github.com/kpot/keras-transformer
'''


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
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
        return input_shape[0],  self.features_dim

def gelu(x):
    """
    GELU activation, described in paper "Gaussian Error Linear Units (GELUs)"
    https://arxiv.org/pdf/1606.08415.pdf
    """
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + K.tanh(c * (x + 0.044715 * K.pow(x, 3))))


class LayerNormalization(Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    "Unlike batch normalization, layer normalization performs exactly
    the same computation at training and test times."
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class TransformerTransition(Layer):
    """
    Transformer transition function. The same function is used both
    in classical in Universal Transformers. Except that in Universal
    Transformer it is also shared between time steps.
    """

    def __init__(self, activation: Union[str, Callable],
                 size_multiplier: int = 4, **kwargs):
        """
        :param activation: activation function. Must be a string or a callable.
        :param size_multiplier: How big the hidden dimension should be.
          Most of the implementation use transition functions having 4 times
          more hidden units than the model itself.
        :param kwargs: Keras-specific layer arguments.
        """
        self.activation = activations.get(activation)
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['activation'] = activations.serialize(self.activation)
        config['size_multiplier'] = self.size_multiplier
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        d_model = input_shape[-1]
        self.weights1 = self.add_weight(
            name='weights1',
            shape=(d_model, self.size_multiplier * d_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases1 = self.add_weight(
            name='biases1',
            shape=(self.size_multiplier * d_model,),
            initializer='zeros',
            trainable=True)
        self.weights2 = self.add_weight(
            name='weights2',
            shape=(self.size_multiplier * d_model, d_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases2 = self.add_weight(
            name='biases2',
            shape=(d_model,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        d_model = input_shape[-1]
        step1 = self.activation(
            K.bias_add(
                K.dot(K.reshape(inputs, (-1, d_model)),
                      self.weights1),
                self.biases1,
                data_format='channels_last'))
        step2 = K.bias_add(
            K.dot(step1, self.weights2),
            self.biases2,
            data_format='channels_last')
        result = K.reshape(step2, (-1,) + input_shape[-2:])
        return result


class TransformerBlock:
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each such block is, essentially:
    - Multi-head self-attention (masked or unmasked, with attention dropout,
      but without input dropout)
    - Residual connection,
    - Dropout
    - Layer normalization
    - Transition function
    - Residual connection
    - Dropout
    - Layer normalization
    Also check TransformerACT class if you need support for ACT (Adaptive
    Computation Time).
    IMPORTANT: The older Transformer 2017 model ("Attention is all you need")
    uses slightly different order of operations. A quote from the paper:
        "We apply dropout [33] to the output of each sub-layer,
         before it is added to the sub-layer input and normalized"
    while the Universal Transformer paper puts dropout one step *after*
    the sub-layers's output was added to its input (Figure 4 in the paper).
    In this code the order from the Universal Transformer is used, as arguably
    more reasonable. You can use classical Transformer's (2017) way of
    connecting the pieces by passing vanilla_wiring=True to the constructor.
    """
    def __init__(self, name: str, num_heads: int,
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'gelu',
                 compression_window_size: int = None,
                 use_masking: bool = True,
                 vanilla_wiring=False):
        self.attention_layer = MultiHeadSelfAttention(
            num_heads, use_masking=use_masking, dropout=attention_dropout,
            compression_window_size=compression_window_size,
            name=f'{name}_self_attention')
        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')
        self.dropout_layer = (
            Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')
        self.transition_layer = TransformerTransition(
            name=f'{name}_transition', activation=activation)
        self.addition_layer = Add(name=f'{name}_add')
        self.vanilla_wiring = vanilla_wiring

    def __call__(self, _input):
        output = self.attention_layer(_input)
        post_residual1 = (
            self.addition_layer([_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.addition_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)
        output = self.transition_layer(norm1_output)
        post_residual2 = (
            self.addition_layer([norm1_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.addition_layer([norm1_output, output])))
        output = self.norm2_layer(post_residual2)
        return output


class TransformerACT(Layer):
    """
    Implements Adaptive Computation Time (ACT) for the Transformer model
    https://arxiv.org/abs/1603.08983
    How to use:
        transformer_depth = 8
        block = TransformerBlock('Transformer', num_heads=8)
        act_layer = TransformerACT()
        next_input = input  # (batch_size, sequence_length, input_size)
        for i in range(transformer_depth):
            next_input = block(next_input, step=i)
            next_input, act_weighted_output = act_layer(next_input)
        act_layer.finalize()  # adds loss
        result = act_weighted_output
    """
    def __init__(self, halt_epsilon=0.01, time_penalty=0.01, **kwargs):
        """
        :param halt_epsilon: a small constant that allows computation to halt
            after a single update (sigmoid never reaches exactly 1.0)
        :param time_penalty: parameter that weights the relative cost
            of computation versus error. The larger it is, the less
            computational steps the network will try to make and vice versa.
            The default value of 0.01 works well for Transformer.
        :param kwargs: Any standard parameters for a layer in Keras (like name)
        """
        self.halt_epsilon = halt_epsilon
        self.time_penalty = time_penalty
        self.ponder_cost = None
        self.weighted_output = None
        self.zeros_like_input = None
        self.zeros_like_halting = None
        self.ones_like_halting = None
        self.halt_budget = None
        self.remainder = None
        self.active_steps = None
        super().__init__(**kwargs)

    def get_config(self):
        return dict(
            super().get_config(),
            halt_epsilon=self.halt_epsilon,
            time_penalty=self.time_penalty)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) == 3
        _, sequence_length, d_model = input_shape
        self.halting_kernel = self.add_weight(
            name='halting_kernel',
            shape=(d_model, 1),
            initializer='glorot_uniform',
            trainable=True)
        self.halting_biases = self.add_weight(
            name='halting_biases',
            shape=(1,),
            initializer=initializers.Constant(0.1),
            trainable=True)
        self.time_penalty_t = K.constant(self.time_penalty, dtype=K.floatx())
        return super().build(input_shape)

    def initialize_control_tensors(self, halting):
        """
        Initializes constants and some step-tracking variables
        during the first call of the layer (since for the Universal Transformer
        all the following calls are supposed to be with inputs of identical
        shapes).
        """
        self.zeros_like_halting = K.zeros_like(
            halting, name='zeros_like_halting')
        self.ones_like_halting = K.ones_like(
            halting, name='ones_like_halting')
        self.remainder = self.ones_like_halting
        self.active_steps = self.zeros_like_halting
        self.halt_budget = self.ones_like_halting - self.halt_epsilon

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        sequence_length, d_model = input_shape[-2:]
        # output of the "sigmoid halting unit" (not the probability yet)
        halting = K.sigmoid(
            K.reshape(
                K.bias_add(
                    K.dot(K.reshape(inputs, [-1, d_model]),
                          self.halting_kernel),
                    self.halting_biases,
                    data_format='channels_last'),
                [-1, sequence_length]))
        if self.zeros_like_halting is None:
            self.initialize_control_tensors(halting)
        # useful flags
        step_is_active = K.greater(self.halt_budget, 0)
        no_further_steps = K.less_equal(self.halt_budget - halting, 0)
        # halting probability is equal to
        # a. halting output if this isn't the last step (we have some budget)
        # b. to remainder if it is,
        # c. and zero for the steps that shouldn't be executed at all
        #    (out of budget for them)
        halting_prob = K.switch(
            step_is_active,
            K.switch(
                no_further_steps,
                self.remainder,
                halting),
            self.zeros_like_halting)
        self.active_steps += K.switch(
            step_is_active,
            self.ones_like_halting,
            self.zeros_like_halting)
        # We don't know which step is the last, so we keep updating
        # expression for the loss with each call of the layer
        self.ponder_cost = (
            self.time_penalty_t * K.mean(self.remainder + self.active_steps))
        # Updating "the remaining probability" and the halt budget
        self.remainder = K.switch(
            no_further_steps,
            self.remainder,
            self.remainder - halting)
        self.halt_budget -= halting  # OK to become negative

        # If none of the inputs are active at this step, then instead
        # of zeroing them out by multiplying to all-zeroes halting_prob,
        # we can simply use a constant tensor of zeroes, which means that
        # we won't even calculate the output of those steps, saving
        # some real computational time.
        if self.zeros_like_input is None:
            self.zeros_like_input = K.zeros_like(
                inputs, name='zeros_like_input')
        # just because K.any(step_is_active) doesn't work in PlaidML
        any_step_is_active = K.greater(
            K.sum(K.cast(step_is_active, 'int32')), 0)
        step_weighted_output = K.switch(
            any_step_is_active,
            K.expand_dims(halting_prob, -1) * inputs,
            self.zeros_like_input)
        if self.weighted_output is None:
            self.weighted_output = step_weighted_output
        else:
            self.weighted_output += step_weighted_output
        return [inputs, self.weighted_output]

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

    def finalize(self):
        self.add_loss(self.ponder_cost)


get_custom_objects().update({
    'LayerNormalization': LayerNormalization,
    'TransformerTransition': TransformerTransition,
    'TransformerACT': TransformerACT,
    'gelu': gelu,
})


class _BaseMultiHeadAttention(Layer):
    """
    Base class for two types of Multi-head attention layers:
    Self-attention and its more general form used in decoders (the one which
    takes values and keys from the encoder).
    """
    def __init__(self, num_heads: int, use_masking: bool,
                 dropout: float = 0.0,
                 compression_window_size: int = None,
                 **kwargs):
        """
        :param num_heads: number of attention heads
        :param use_masking: when True, forbids the attention to see the further
          elements in the sequence (particularly important in language
          modelling).
        :param dropout: dropout that should be applied to the attention
          (after the softmax).
        :param compression_window_size: an integer value >= 1 controlling
          how much we should compress the attention. For more details,
          read about memory-compressed self-attention in
          "Generating Wikipedia by summarizing long sequences"
          (https://arxiv.org/pdf/1801.10198.pdf).
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.dropout = dropout
        if (compression_window_size is not None
                and compression_window_size <= 0):
            assert ValueError(
                f"Too small compression window ({compression_window_size})")
        self.compression_window_size = compression_window_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['use_masking'] = self.use_masking
        config['dropout'] = self.dropout
        config['compression_window_size'] = self.compression_window_size
        return config

    # noinspection PyAttributeOutsideInit
    def build_output_params(self, d_model):
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True)
        if self.compression_window_size is not None:
            self.k_conv_kernel = self.add_weight(
                name='k_conv_kernel',
                shape=(self.compression_window_size,
                       d_model // self.num_heads,
                       d_model // self.num_heads),
                initializer='glorot_uniform',
                trainable=True)
            self.k_conv_bias = self.add_weight(
                name='k_conv_bias',
                shape=(d_model // self.num_heads,),
                initializer='zeros',
                trainable=True)
            self.v_conv_kernel = self.add_weight(
                name='v_conv_kernel',
                shape=(self.compression_window_size,
                       d_model // self.num_heads,
                       d_model // self.num_heads),
                initializer='glorot_uniform',
                trainable=True)
            self.v_conv_bias = self.add_weight(
                name='v_conv_bias',
                shape=(d_model // self.num_heads,),
                initializer='zeros',
                trainable=True)

    def validate_model_dimensionality(self, d_model: int):
        if d_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')

    def attention(self, pre_q, pre_v, pre_k, out_seq_len: int, d_model: int,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param out_seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        if self.compression_window_size is None:
            k_transposed = K.permute_dimensions(pre_k, [0, 2, 3, 1])
        else:
            # Memory-compressed attention described in paper
            # "Generating Wikipedia by Summarizing Long Sequences"
            # (https://arxiv.org/pdf/1801.10198.pdf)
            # It compresses keys and values using 1D-convolution which reduces
            # the size of Q * K_transposed from roughly seq_len^2
            # to convoluted_seq_len^2. If we use strided convolution with
            # window size = 3 and stride = 3, memory requirements of such
            # memory-compressed attention will be 9 times smaller than
            # that of the original version.
            if self.use_masking:
                raise NotImplementedError(
                    "Masked memory-compressed attention has not "
                    "been implemented yet")
            k = K.permute_dimensions(pre_k, [0, 2, 1, 3])
            k, v = [
                K.reshape(
                    # Step 3: Return the result to its original dimensions
                    # (batch_size, num_heads, seq_len, d_model//heads)
                    K.bias_add(
                        # Step 3: ... and add bias
                        K.conv1d(
                            # Step 2: we "compress" K and V using strided conv
                            K.reshape(
                                # Step 1: we reshape K and V to
                                # (batch + num_heads,  seq_len, d_model//heads)
                                item,
                                (-1,
                                 K.int_shape(item)[-2],
                                 d_model // self.num_heads)),
                            kernel,
                            strides=self.compression_window_size,
                            padding='valid', data_format='channels_last'),
                        bias,
                        data_format='channels_last'),
                    # new shape
                    K.concatenate([
                        K.shape(item)[:2],
                        [-1, d_model // self.num_heads]]))
                for item, kernel, bias in (
                    (k, self.k_conv_kernel, self.k_conv_bias),
                    (v, self.v_conv_kernel, self.v_conv_bias))]
            k_transposed = K.permute_dimensions(k, [0, 1, 3, 2])
        # shaping K into (batch_size, num_heads, d_model//heads, seq_len)
        # for further matrix multiplication
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads),
                            dtype=K.floatx())
        q_shape = K.int_shape(q)
        k_t_shape = K.int_shape(k_transposed)
        v_shape = K.int_shape(v)
        # before performing batch_dot all tensors are being converted to 3D
        # shape (batch_size * num_heads, rows, cols) to make sure batch_dot
        # performs identically on all backends
        attention_heads = K.reshape(
            K.batch_dot(
                self.apply_dropout_if_needed(
                    K.softmax(
                        self.mask_attention_if_needed(
                            K.batch_dot(
                                K.reshape(q, (-1,) + q_shape[-2:]),
                                K.reshape(k_transposed,
                                          (-1,) + k_t_shape[-2:]))
                            / sqrt_d)),
                    training=training),
                K.reshape(v, (-1,) + v_shape[-2:])),
            (-1, self.num_heads, q_shape[-2], v_shape[-1]))
        attention_heads_merged = K.reshape(
            K.permute_dimensions(attention_heads, [0, 2, 1, 3]),
            (-1, d_model))
        attention_out = K.reshape(
            K.dot(attention_heads_merged, self.output_weights),
            (-1, out_seq_len, d_model))
        return attention_out

    def apply_dropout_if_needed(self, attention_softmax, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(attention_softmax, self.dropout)

            return K.in_train_phase(dropped_softmax, attention_softmax,
                                    training=training)
        return attention_softmax

    def mask_attention_if_needed(self, dot_product):
        """
        Makes sure that (when enabled) each position
        (of a decoder's self-attention) cannot attend to subsequent positions.
        This is achieved by assigning -inf (or some large negative number)
        to all invalid connections. Later softmax will turn them into zeros.
        We need this to guarantee that decoder's predictions are based
        on what has happened before the position, not after.
        The method does nothing if masking is turned off.
        :param dot_product: scaled dot-product of Q and K after reshaping them
        to 3D tensors (batch * num_heads, rows, cols)
        """
        if not self.use_masking:
            return dot_product
        last_dims = K.int_shape(dot_product)[-2:]
        low_triangle_ones = (
            np.tril(np.ones(last_dims))
            # to ensure proper broadcasting
            .reshape((1,) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (
            K.constant(low_triangle_ones, dtype=K.floatx()) * dot_product +
            K.constant(close_to_negative_inf * inverse_low_triangle))
        return result


class MultiHeadAttention(_BaseMultiHeadAttention):
    """
    Multi-head attention which can use two inputs:
    First: from the encoder - it's used to project the keys and the values
    Second: from the decoder - used to project the queries.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise ValueError(
                'You must call this layer passing a list of two tensors'
                '(for keys/values and queries)')
        values_dim, query_dim = input_shape[0][-1], input_shape[1][-1]
        if query_dim != values_dim:
            raise ValueError(
                f'Both keys/value and query inputs must be '
                f'of the same dimensionality, instead of '
                f'{values_dim} and {query_dim}.')
        d_model = query_dim
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_k and W_v which
        # are, in turn, concatenated W matrices of keys, and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.kv_weights = self.add_weight(
            name='kv_weights', shape=(d_model, d_model * 2),
            initializer='glorot_uniform', trainable=True)
        self.q_weights = self.add_weight(
            name='q_weights', shape=(d_model, d_model),
            initializer='glorot_uniform', trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not (isinstance(inputs, list) and len(inputs) == 2):
            raise ValueError(
                'You can call this layer only with a list of two tensors '
                '(for keys/values and queries)')
        key_values_input, query_input = inputs
        _, value_seq_len, d_model = K.int_shape(key_values_input)
        query_seq_len = K.int_shape(inputs[1])[-2]
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        kv = K.dot(K.reshape(key_values_input, [-1, d_model]), self.kv_weights)
        # splitting the keys, the values and the queries before further
        # processing
        pre_k, pre_v = [
            K.reshape(
                # K.slice(kv, (0, i * d_model), (-1, d_model)),
                kv[:, i * d_model: (i + 1) * d_model],
                (-1, value_seq_len,
                 self.num_heads, d_model // self.num_heads))
            for i in range(2)]
        pre_q = K.reshape(
            K.dot(K.reshape(query_input, [-1, d_model]), self.q_weights),
            (-1, query_seq_len, self.num_heads, d_model // self.num_heads))
        return self.attention(pre_q, pre_v, pre_k, query_seq_len, d_model,
                              training=kwargs.get('training'))


class MultiHeadSelfAttention(_BaseMultiHeadAttention):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input and has implementation which is better suited for
    such use case that more general MultiHeadAttention class.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')
        d_model = input_shape[-1]
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_q, W_k and W_v which
        # are, in turn, concatenated W matrices of keys, queries and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_uniform',
            trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = K.int_shape(inputs)
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        qkv = K.dot(K.reshape(inputs, [-1, d_model]), self.qkv_weights)
        # splitting the keys, the values and the queries before further
        # processing
        pre_q, pre_k, pre_v = [
            K.reshape(
                # K.slice(qkv, (0, i * d_model), (-1, d_model)),
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model,
                                       training=kwargs.get('training'))
        return attention_out

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'MultiHeadAttention': MultiHeadAttention,
})



def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


class AddCoordinateEncoding(AddPositionalEncoding):
    """
    Implements coordinate encoding described in section 2.1
    of "Universal Transformers" (https://arxiv.org/abs/1807.03819).
    In other words, injects two signals at once: current position in
    the sequence, and current step (vertically) in the transformer model.
    """

    def build(self, input_shape):
        super().build(input_shape)
        _, length, hidden_size = input_shape

    def call(self, inputs, step=None, **kwargs):
        if step is None:
            raise ValueError("Please, provide current Transformer's step"
                             "using 'step' keyword argument.")
        pos_encoded_added = super().call(inputs, **kwargs)
        step_signal = K.expand_dims(self.signal[:, step, :], axis=1)
        return pos_encoded_added + step_signal


class TransformerCoordinateEmbedding(Layer):
    """
    Represents trainable positional embeddings for the Transformer model:
    1. word position embeddings - one for each position in the sequence.
    2. depth embeddings - one for each block of the model
    Calling the layer with the Transformer's input will return a new input
    with those embeddings added.
    """

    def __init__(self, max_transformer_depth: int, **kwargs):
        self.max_depth = max_transformer_depth
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['max_transformer_depth'] = self.max_depth
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        sequence_length, d_model = input_shape[-2:]
        self.word_position_embeddings = self.add_weight(
            shape=(sequence_length, d_model),
            initializer='uniform',
            name='word_position_embeddings',
            trainable=True)
        self.depth_embeddings = self.add_weight(
            shape=(self.max_depth, d_model),
            initializer='uniform',
            name='depth_position_embeddings',
            trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        depth = kwargs.get('step')
        if depth is None:
            raise ValueError("Please, provide current Transformer's step"
                             "using 'step' keyword argument.")
        result = inputs + self.word_position_embeddings
        if depth is not None:
            result = result + self.depth_embeddings[depth]
        return result


get_custom_objects().update({
    'TransformerCoordinateEmbedding': TransformerCoordinateEmbedding,
    'AddCoordinateEncoding': AddCoordinateEncoding,
    'AddPositionalEncoding': AddCoordinateEncoding,
})

########################################################










def model_7_11_wgd_v0(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,b_embeddings_weight,class_num):
    print("model_7_11_wgd_v0 : lstm +diff dim +concat + trainable adv pid")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    b_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid_iid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)
    b_embedding = Embedding(
        name="b_embedding",
        input_dim=b_embeddings_weight.shape[0],
        weights=[b_embeddings_weight],
        output_dim=b_embeddings_weight.shape[1],
        trainable=True)


#     tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
    lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()
    # gru_layer_2 = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))




    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)
    b_embed = b_embedding(b_ids)

    embed = concatenate([x_embed, y_embed,z_embed,a_embed,b_embed],axis=-1)


    x_emb = sdrop(embed)
    lstm1 = lstm_layer(x_emb)
    gru1 = gru_layer(lstm1)
    att_1 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(lstm1)

#     features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),tf_idf_embedding,countvec_embedding])  
    features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),countvec_embedding])  

    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,countvec], outputs=[output_age,output_gender])
    return model



# 改share特征提取
def model_7_11_wgd_v1(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,class_num):
    print("model_7_11_wgd_v1 : 4id-share")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)

    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
    lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()


    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)

    x_embed = sdrop(x_embed)
    y_embed = sdrop(y_embed)
    z_embed = sdrop(z_embed)
    a_embed = sdrop(a_embed)

    x_embed = lstm_layer(x_embed)
    y_embed = lstm_layer(y_embed)
    z_embed = lstm_layer(z_embed)
    a_embed = lstm_layer(a_embed)

    lstm1 = concatenate([x_embed, y_embed,z_embed,a_embed],axis=-1)


    gru1 = gru_layer(lstm1)
    att_1 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(lstm1)

    features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),countvec_embedding])  

    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,countvec], outputs=[output_age,output_gender])
    return model



# 改我的backbone
def model_7_11_wgd_v2(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,b_embeddings_weight,class_num):
    print("model_7_11_wgd_v2 : my_backbobe")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    b_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid_iid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)
    b_embedding = Embedding(
        name="b_embedding",
        input_dim=b_embeddings_weight.shape[0],
        weights=[b_embeddings_weight],
        output_dim=b_embeddings_weight.shape[1],
        trainable=True)


#     tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
    lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))

    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)
    b_embed = b_embedding(b_ids)

    embed = concatenate([x_embed, y_embed,z_embed,a_embed,b_embed],axis=-1)


    x_emb = sdrop(embed)
    lstm1 = lstm_layer(x_emb)
    gru1 = gru_layer(lstm1)

    semantic = TimeDistributed(Dense(100, activation="relu"))(gru1)
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(100,))(semantic)
    semantic_attn = AttentionWeightedAverage()(semantic)

 
    features = concatenate([pool_rnn, semantic_attn,countvec_embedding])


    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,countvec], outputs=[output_age,output_gender])
    return model





# 双层lstm l2
def model_7_11_wgd_v3(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,b_embeddings_weight,class_num):
    print("model_7_11_wgd_v3 : 双层lstm")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    b_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid_iid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)
    b_embedding = Embedding(
        name="b_embedding",
        input_dim=b_embeddings_weight.shape[0],
        weights=[b_embeddings_weight],
        output_dim=b_embeddings_weight.shape[1],
        trainable=True)


#     tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
#     lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123),kernel_regularizer=regularizers.l2(0.001)))
#     gru_layer = Bidirectional(CuDNNLSTM(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123),kernel_regularizer=regularizers.l2(0.001)))
#     cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform",kernel_regularizer=regularizers.l2(0.001))
#     max_pool = GlobalMaxPooling1D()
    lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNLSTM(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
    max_pool = GlobalMaxPooling1D()

    # gru_layer_2 = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))




    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)
    b_embed = b_embedding(b_ids)

    embed = concatenate([x_embed, y_embed,z_embed,a_embed,b_embed],axis=-1)


    x_emb = sdrop(embed)
    lstm1 = lstm_layer(x_emb)
    gru1 = gru_layer(lstm1)
    att_1 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(lstm1)
    #fn1 = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(100)(lstm1))))
    #x1 = concatenate([att_1,att_2,fn1,max_pool(gru1)])
    # x1 = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1)])
    # 修改，dence维度增加至1200

#     features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),tf_idf_embedding,countvec_embedding])  
    features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),countvec_embedding])  

    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,countvec], outputs=[output_age,output_gender])
    return model



# 加atten
def model_7_11_wgd_v4(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,b_embeddings_weight,class_num):
    print("model_7_11_wgd_v4 : +attn")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    b_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid_iid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)
    b_embedding = Embedding(
        name="b_embedding",
        input_dim=b_embeddings_weight.shape[0],
        weights=[b_embeddings_weight],
        output_dim=b_embeddings_weight.shape[1],
        trainable=True)


#     tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
    lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
#     avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()
    # gru_layer_2 = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))




    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)
    b_embed = b_embedding(b_ids)

    embed = concatenate([x_embed, y_embed,z_embed,a_embed,b_embed],axis=-1)


    x_emb = sdrop(embed)
    lstm1 = lstm_layer(x_emb)
    att_1 = AttentionLayer(attention_size=190)(lstm1)
    gru1 = gru_layer(lstm1)
    att_2 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(lstm1)

#     features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),tf_idf_embedding,countvec_embedding])  
    features = concatenate([att_1,att_2,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),countvec_embedding])  

    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,countvec], outputs=[output_age,output_gender])
    return model



# 双层GRU
def model_7_11_wgd_v5(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,b_embeddings_weight,class_num):
    print("model_7_11_wgd_v5 : 双层GRU")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    b_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid_iid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)
    b_embedding = Embedding(
        name="b_embedding",
        input_dim=b_embeddings_weight.shape[0],
        weights=[b_embeddings_weight],
        output_dim=b_embeddings_weight.shape[1],
        trainable=True)


#     tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
    lstm_layer = Bidirectional(CuDNNGRU(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()

    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)
    b_embed = b_embedding(b_ids)

    embed = concatenate([x_embed, y_embed,z_embed,a_embed,b_embed],axis=-1)


    x_emb = sdrop(embed)
    lstm1 = lstm_layer(x_emb)
    gru1 = gru_layer(lstm1)
    att_1 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(lstm1)
    #fn1 = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(100)(lstm1))))
    #x1 = concatenate([att_1,att_2,fn1,max_pool(gru1)])
    # x1 = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1)])
    # 修改，dence维度增加至1200

#     features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),tf_idf_embedding,countvec_embedding])  
    features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),countvec_embedding])  

    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,countvec], outputs=[output_age,output_gender])
    return model

def model_trans_v1(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,class_num):
    print("model_trans_v1 ")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
#     tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=True)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

#     tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))


    dropout_param = 0.3
    
    sdrop = SpatialDropout1D(rate=dropout_param)
    transformer_block = TransformerBlock(
        name='transformer',
        num_heads=6,
        residual_dropout=0.1,
        attention_dropout=0.1,
        use_masking=True)

    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
    max_pool = GlobalMaxPooling1D()

    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)


    embed = concatenate([x_embed, y_embed,z_embed],axis=-1)


    x_emb = sdrop(embed)
    trans = transformer_block(x_emb)
    gru1 = gru_layer(trans)
    att_1 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(trans)

    features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(trans),countvec_embedding])      
    
    
    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
    
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
    model = Model(inputs=[x_ids,y_ids,z_ids,countvec], outputs=[output_age,output_gender])
    return model



def model_7_11_wgd_v6(sent_length, x_embeddings_weight,y_embeddings_weight,z_embeddings_weight,a_embeddings_weight,b_embeddings_weight,class_num):
    print("model_7_11_wgd_v6 : lstm +diff dim +concat + trainable adv pid")
    x_ids = Input(shape=(sent_length,), dtype='int32')
    y_ids = Input(shape=(sent_length,), dtype='int32')
    z_ids = Input(shape=(sent_length,), dtype='int32')
    a_ids = Input(shape=(sent_length,), dtype='int32')
    b_ids = Input(shape=(sent_length,), dtype='int32')
    tf_idf = Input(shape=(sent_length,), dtype='float32')
    countvec = Input(shape=(12,), dtype='float32')

    # aid_cid_adv_id_pid_iid
    x_embedding = Embedding(
        name="x_embedding",
        input_dim=x_embeddings_weight.shape[0],
        weights=[x_embeddings_weight],
        output_dim=x_embeddings_weight.shape[1],
        trainable=False)
    y_embedding = Embedding(
        name="y_embedding",
        input_dim=y_embeddings_weight.shape[0],
        weights=[y_embeddings_weight],
        output_dim=y_embeddings_weight.shape[1],
        trainable=False)
    z_embedding = Embedding(
        name="z_embedding",
        input_dim=z_embeddings_weight.shape[0],
        weights=[z_embeddings_weight],
        output_dim=z_embeddings_weight.shape[1],
        trainable=True)

    a_embedding = Embedding(
        name="a_embedding",
        input_dim=a_embeddings_weight.shape[0],
        weights=[a_embeddings_weight],
        output_dim=a_embeddings_weight.shape[1],
        trainable=True)
    b_embedding = Embedding(
        name="b_embedding",
        input_dim=b_embeddings_weight.shape[0],
        weights=[b_embeddings_weight],
        output_dim=b_embeddings_weight.shape[1],
        trainable=True)


    tf_idf_embedding = Activation(activation="relu")(BatchNormalization()(Dense(200)(tf_idf)))
    countvec_embedding = Activation(activation="relu")(BatchNormalization()(Dense(20)(countvec)))

    dropout_param = 0.3



    sdrop = SpatialDropout1D(rate=dropout_param)
    lstm_layer = Bidirectional(CuDNNLSTM(200, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed = 123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")
#     avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()



    x_embed = x_embedding(x_ids)
    y_embed = y_embedding(y_ids)
    z_embed = z_embedding(z_ids)
    a_embed = a_embedding(a_ids)
    b_embed = b_embedding(b_ids)

    embed = concatenate([x_embed, y_embed,z_embed,a_embed,b_embed],axis=-1)


    x_emb = sdrop(embed)
    lstm1 = lstm_layer(x_emb)
    gru1 = gru_layer(lstm1)
    att_1 = AttentionLayer(attention_size=190)(gru1)
    cnn1 = cnn1d_layer(lstm1)

    features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),tf_idf_embedding,countvec_embedding])  
#     features = concatenate([att_1,max_pool(cnn1),max_pool(gru1),max_pool(lstm1),countvec_embedding])  

    x1 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(1200)(features))))
    x2 = Dropout(dropout_param)(Activation(activation="relu")(BatchNormalization()(Dense(800)(x1))))
    x3 = Activation(activation="relu")(BatchNormalization()(Dense(500)(x2)))
    output_age = Dense(10, activation="softmax",name='output_age')(x3)
    output_gender= Dense(2, activation="softmax",name='output_gener')(x3)
    model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,tf_idf,countvec], outputs=[output_age,output_gender])
#     model = Model(inputs=[x_ids,y_ids,z_ids,a_ids,b_ids,countvec], outputs=[output_age,output_gender])
    return model