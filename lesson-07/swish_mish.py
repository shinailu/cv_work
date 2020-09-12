from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
import keras.backend as K
def mish(x):
    return x * K.tanh(K.softplus(x))
def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass
    return x * K.sigmoid(x)
def _cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4
    eps = 1.1e-5
    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation(mish)(x)
    x = Conv2D(_nb_filter, (3, 3), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(x)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation(swish)(x)
    return x
input = Input(shape=(32, 280, 1), name='the_input')
y_pred=_cnn(input, 5000)
basemodel = Model(inputs=input, outputs=y_pred)
basemodel.summary()

