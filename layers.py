from keras.engine.topology import Layer
from keras import backend as K
import itertools

class Normalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG and GAN networks.
    '''

    def __init__(self, type="vgg", value=120, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.type = type
        self.value = value

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if self.type == "gan":
            return (x - self.value) / self.value # [0, 255] -> [-1, +1]
        else:
            if K.backend() == "theano":
                import theano.tensor as T
                x = T.set_subtensor(x[:, 0, :, :], x[:, 0, :, :] - 103.939)
                x = T.set_subtensor(x[:, 1, :, :], x[:, 1, :, :] - 116.779)
                x = T.set_subtensor(x[:, 2, :, :], x[:, 2, :, :] - 123.680)
            else:
                # No exact substitute for set_subtensor in tensorflow
                # So we subtract an approximate value
                x = x - self.value
            return x


    def get_output_shape_for(self, input_shape):
        return input_shape


class Denormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG and GAN networks.
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return (x + 1) * 127.5

    def get_output_shape_for(self, input_shape):
        return input_shape


''' Theano Backend function '''
def depth_to_scale_th(input, scale, channels):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''
    import theano.tensor as T

    b, k, row, col = input.shape
    output_shape = (b, channels, row * scale, col * scale)

    out = T.zeros(output_shape)
    r = scale

    for y, x in itertools.product(range(scale), repeat=2):
        out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x :: r * r, :, :])

    return out


''' Tensorflow Backend Function '''
def depth_to_scale_tf(input, scale, channels):
    try:
        import tensorflow as tf
    except ImportError:
        print("Could not import Tensorflow for depth_to_scale operation. Please install Tensorflow or switch to Theano backend")
        exit()

    def _phase_shift(I, r):
        ''' Function copied as is from https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py'''

        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a * r, b * r, 1))

    if channels > 1:
        Xc = tf.split(3, 3, input)
        X = tf.concat(3, [_phase_shift(x, scale) for x in Xc])
    else:
        X = _phase_shift(input, scale)
    return X

'''
Implementation is incomplete. Use lambda layer for now.
'''

class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if K.backend() == "theano":
            y = depth_to_scale_th(x, self.r, self.channels)
        else:
            y = depth_to_scale_tf(x, self.r, self.channels)
        return y

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, self.channels, r * self.r, c * self.r)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, self.channels)