from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import get_file

from keras_ops import fit as bypass_fit, smooth_gan_labels
from sklearn.feature_extraction.image import extract_patches_2d

from layers import Normalize, Denormalize, SubPixelUpscaling
from loss import AdversarialLossRegularizer, ContentVGGRegularizer, TVRegularizer, psnr, dummy_loss

import os
import time
import h5py
import numpy as np
import json
from scipy.misc import imresize, imsave
from scipy.ndimage.filters import gaussian_filter

THEANO_WEIGHTS_PATH_NO_TOP = r'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = r"https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

learning_rate = 1e-4

if not os.path.exists("weights/"):
    os.makedirs("weights/")

if not os.path.exists("val_images/"):
    os.makedirs("val_images/")

if K.image_dim_ordering() == "th":
    channel_axis = 1
else:
    channel_axis = -1

class VGGNetwork:
    '''
    Helper class to load VGG and its weights to the FastNet model
    '''

    def __init__(self, img_width=28, img_height=28, vgg_weight=1.0):
        self.img_height = img_height
        self.img_width = img_width
        self.vgg_weight = vgg_weight

        self.vgg_layers = None

    def append_vgg_network(self, x_in, true_X_input, pre_train=False):

        # Append the initial inputs to the outputs of the DeblurResNet
        x = merge([x_in, true_X_input], mode='concat', concat_axis=0)

        # Normalize the inputs via custom VGG Normalization layer
        x = Normalize(name="normalize_vgg")(x)

        # Begin adding the VGG layers
        x = Convolution2D(64, 3, 3, activation='relu', name='vgg_conv1_1', border_mode='same')(x)

        x = Convolution2D(64, 3, 3, activation='relu', name='vgg_conv1_2', border_mode='same')(x)
        x = MaxPooling2D(name='vgg_maxpool1')(x)

        x = Convolution2D(128, 3, 3, activation='relu', name='vgg_conv2_1', border_mode='same')(x)

        if pre_train:
            vgg_regularizer2 = ContentVGGRegularizer(weight=self.vgg_weight)
            x = Convolution2D(128, 3, 3, activation='relu', name='vgg_conv2_2', border_mode='same',
                              activity_regularizer=vgg_regularizer2)(x)
        else:
            x = Convolution2D(128, 3, 3, activation='relu', name='vgg_conv2_2', border_mode='same')(x)
        x = MaxPooling2D(name='vgg_maxpool2')(x)

        x = Convolution2D(256, 3, 3, activation='relu', name='vgg_conv3_1', border_mode='same')(x)
        x = Convolution2D(256, 3, 3, activation='relu', name='vgg_conv3_2', border_mode='same')(x)

        x = Convolution2D(256, 3, 3, activation='relu', name='vgg_conv3_3', border_mode='same')(x)
        x = MaxPooling2D(name='vgg_maxpool3')(x)

        x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv4_1', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv4_2', border_mode='same')(x)

        x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv4_3', border_mode='same')(x)
        x = MaxPooling2D(name='vgg_maxpool4')(x)

        x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv5_1', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv5_2', border_mode='same')(x)

        if not pre_train:
            vgg_regularizer5 = ContentVGGRegularizer(weight=self.vgg_weight)
            x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv5_3', border_mode='same',
                          activity_regularizer=vgg_regularizer5)(x)
        else:
            x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv5_3', border_mode='same')(x)
        x = MaxPooling2D(name='vgg_maxpool5')(x)

        return x

    def load_vgg_weight(self, model):
        # Loading VGG 16 weights
        if K.image_dim_ordering() == "th":
            weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models')
        else:
            weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models')
        f = h5py.File(weights)

        layer_names = [name for name in f.attrs['layer_names']]

        if self.vgg_layers is None:
            self.vgg_layers = [layer for layer in model.layers
                               if 'vgg_' in layer.name]

        for i, layer in enumerate(self.vgg_layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        # Freeze all VGG layers
        for layer in self.vgg_layers:
            layer.trainable = False

        return model


class DiscriminatorNetwork:

    def __init__(self, img_width=28, img_height=28, adversarial_loss_weight=1, small_model=False):
        self.img_width = img_width
        self.img_height = img_height
        self.adversarial_loss_weight = adversarial_loss_weight
        self.small_model = small_model

        self.k = 3
        self.mode = 2
        self.weights_path = "weights/Discriminator weights.h5"

        self.gan_layers = None

    def append_gan_network(self, true_X_input):

        # Normalize the inputs via custom VGG Normalization layer
        x = Normalize(type="gan", value=127.5, name="gan_normalize")(true_X_input)

        x = Convolution2D(64, self.k, self.k, border_mode='same', name='gan_conv1_1')(x)
        x = LeakyReLU(0.3, name="gan_lrelu1_1")(x)

        x = Convolution2D(64, self.k, self.k, border_mode='same', name='gan_conv1_2', subsample=(2, 2))(x)
        x = BatchNormalization(mode=self.mode, axis=channel_axis, name='gan_batchnorm1_1')(x)
        x = LeakyReLU(0.3, name='gan_lrelu1_2')(x)

        filters = [128, 256] if self.small_model else [128, 256, 512]

        for i, nb_filters in enumerate(filters):
            for j in range(2):
                subsample = (2, 2) if j == 1 else (1, 1)

                x = Convolution2D(nb_filters, self.k, self.k, border_mode='same', subsample=subsample,
                                  name='gan_conv%d_%d' % (i + 2, j + 1))(x)
                x = BatchNormalization(mode=self.mode, axis=channel_axis, name='gan_batchnorm%d_%d' % (i + 2, j + 1))(x)
                x = LeakyReLU(0.3, name='gan_lrelu_%d_%d' % (i + 2, j + 1))(x)

        x = Flatten(name='gan_flatten')(x)

        output_dim = 128 if self.small_model else 1024

        x = Dense(output_dim, name='gan_dense1')(x)
        x = LeakyReLU(0.3, name='gan_lrelu5')(x)

        gan_regulrizer = AdversarialLossRegularizer(weight=self.adversarial_loss_weight)
        x = Dense(2, activation="softmax", activity_regularizer=gan_regulrizer, name='gan_output')(x)

        return x

    def set_trainable(self, model, value=True):
        if self.gan_layers is None:
            disc_model = [layer for layer in model.layers
                          if 'model' in layer.name][0] # Only disc model is an inner model

            self.gan_layers = [layer for layer in disc_model.layers
                               if 'gan_' in layer.name]

        for layer in self.gan_layers:
            layer.trainable = value

    def load_gan_weights(self, model):
        f = h5py.File(self.weights_path)

        layer_names = [name for name in f.attrs['layer_names']]
        layer_names = layer_names[1:] # First is an input layer. Not needed.

        if self.gan_layers is None:
            self.gan_layers = [layer for layer in model.layers
                                if 'gan_' in layer.name]

        for i, layer in enumerate(self.gan_layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        print("GAN Model weights loaded.")
        return model

    def save_gan_weights(self, model):
        print('GAN Weights are being saved.')
        model.save_weights(self.weights_path, overwrite=True)
        print('GAN Weights saved.')


class GenerativeNetwork:

    def __init__(self, img_width=64, img_height=64, batch_size=16, small_model=False,
                 content_weight=1, tv_weight=2e5, gen_channels=64):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.small_model = small_model

        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.filters = gen_channels
        self.mode = 2
        self.init = 'glorot_uniform'

        self.deblur_res_layers = None

        self.output_func = None

    def create_deblur_model(self, ip):
        x = AveragePooling2D((4, 4), strides=(4, 4), name='db_res_pool2', border_mode='same')(ip)
        x = Convolution2D(self.filters, 5, 5, activation='linear', border_mode='same', name='db_res_conv1',
                          init=self.init)(x)
        x = BatchNormalization(axis=channel_axis, mode=self.mode, name='db_res_bn_1')(x)
        x = LeakyReLU(alpha=0.25, name='db_res_lr1')(x)

        nb_residual = 5 if self.small_model else 15

        for i in range(nb_residual):
            x = self._residual_block(x, i + 1)

        scale = 2 ** self.nb_scales
        tv_regularizer = TVRegularizer(img_width=self.img_width * scale, img_height=self.img_height * scale,
                                       weight=self.tv_weight) #self.tv_weight)

        x = Convolution2D(3, 5, 5, activation='tanh', border_mode='same', activity_regularizer=tv_regularizer,
                          init=self.init, name='db_res_conv_final')(x)

        x = Denormalize(name='db_res_conv_denorm')(x)

        return x

    def _residual_block(self, ip, id):
        init = ip

        x = Convolution2D(self.filters, 3, 3, activation='linear', border_mode='same', name='db_res_conv_' + str(id) + '_1',
                          init=self.init)(ip)
        x = BatchNormalization(axis=channel_axis, mode=self.mode, name='db_res_bn_' + str(id) + '_1')(x)
        x = LeakyReLU(alpha=0.25, name="db_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.filters, 3, 3, activation='linear', border_mode='same', name='db_res_conv_' + str(id) + '_2',
                          init=self.init)(x)
        x = BatchNormalization(axis=channel_axis, mode=self.mode, name='db_res_bn_' + str(id) + '_2')(x)

        m = merge([x, init], mode='sum', name="db_res_merge_" + str(id))

        return m

    def set_trainable(self, model, value=True):
        if self.deblur_res_layers is None:
            self.deblur_res_layers = [layer for layer in model.layers
                                    if 'db_res_' in layer.name]

        for layer in self.deblur_res_layers:
            layer.trainable = value

    def get_generator_output(self, input_img, deblurgan_model):
        if self.output_func is None:
            gen_output_layer = [layer for layer in deblurgan_model.layers
                                if layer.name == "db_res_conv_denorm"][0]
            self.output_func = K.function([deblurgan_model.layers[0].input],
                                          [gen_output_layer.output])

        return self.output_func([input_img])

class GenerativeNetworkFiner:
    
    def __init__(self, img_width=64, img_height=64, batch_size=16, small_model=False,
                 content_weight=1, tv_weight=2e5, gen_channels=64,finest=False):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.small_model = small_model
        
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        
        self.filters = gen_channels
        self.mode = 2
        self.init = 'glorot_uniform'
        
        self.deblur_res_layers = None
        
        self.output_func = None
        self.finest = finest
    
    def create_deblur_model(self, ip, auxIp):
        id = 0
        if self.finest == False:
            ip = AveragePooling2D((2, 2), strides=(2, 2), name='db_res_pool1', border_mode='same')(ip)
            id = 1
        auxIp = UpSampling2D(size=(2, 2),name='db_res_upscale_' + str(id))(auxIp)
        x = merge([ip, auxIp], mode='concat', concat_axis=1)
        x = Convolution2D(self.filters, 5, 5, activation='linear', border_mode='same', name='db_res_conv1',
                          init=self.init)(x)
        x = BatchNormalization(axis=channel_axis, mode=self.mode, name='db_res_bn_1')(x)
        x = LeakyReLU(alpha=0.25, name='db_res_lr1')(x)
                          
        nb_residual = 5 if self.small_model else 15
      
        for i in range(nb_residual):
            x = self._residual_block(x, i + 1)
      
        scale = 2 ** self.nb_scales
        tv_regularizer = TVRegularizer(img_width=self.img_width * scale, img_height=self.img_height * scale, weight=self.tv_weight) #self.tv_weight)
              
        x = Convolution2D(3, 5, 5, activation='tanh', border_mode='same', activity_regularizer=tv_regularizer, init=self.init, name='db_res_conv_final')(x)
                  
        x = Denormalize(name='db_res_conv_denorm')(x)
                      
        return x

    def _residual_block(self, ip, id):
        init = ip
        
        x = Convolution2D(self.filters, 3, 3, activation='linear', border_mode='same', name='db_res_conv_' + str(id) + '_1', init=self.init)(ip)
        x = BatchNormalization(axis=channel_axis, mode=self.mode, name='db_res_bn_' + str(id) + '_1')(x)
        x = LeakyReLU(alpha=0.25, name="db_res_activation_" + str(id) + "_1")(x)
                          
        x = Convolution2D(self.filters, 3, 3, activation='linear', border_mode='same', name='db_res_conv_' + str(id) + '_2', init=self.init)(x)
        x = BatchNormalization(axis=channel_axis, mode=self.mode, name='db_res_bn_' + str(id) + '_2')(x)
                          
        m = merge([x, init], mode='sum', name="db_res_merge_" + str(id))
                          
        return m
    
    def set_trainable(self, model, value=True):
        if self.deblur_res_layers is None:
            self.deblur_res_layers = [layer for layer in model.layers if 'db_res_' in layer.name]
        
        for layer in self.deblur_res_layers:
            layer.trainable = value
    
    def get_generator_output(self, input_img, deblurgan_model):
        if self.output_func is None:
            gen_output_layer = [layer for layer in deblurgan_model.layers if layer.name == "deblur_res_conv_denorm"][0]
        self.output_func = K.function([deblurgan_model.layers[0].input], [gen_output_layer.output])
                            
        return self.output_func([input_img])


class DeblurGANNetwork:

    def __init__(self, img_width=64, img_height=64, batch_size=16, nb_scales=2):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.nb_scales = nb_scales

        self.discriminative_network = None # type: DiscriminatorNetwork
        self.generative_network = None # type: GenerativeNetwork
        self.vgg_network = None # type: VGGNetwork

        self.deblurgan_model_ = None # type: Model
        self.generative_model_ = None # type: Model
        self.discriminative_model_ = None #type: Model


    def build_deblurgan_model(self, use_small_deblurgan=False, use_small_discriminator=False):

        self.generative_network = GenerativeNetwork(self.img_width, self.img_height, self.batch_size, small_model=use_small_deblurgan)
                                                    
        self.generative_network_finer = GenerativeNetworkFiner(self.img_width, self.img_height, self.batch_size, self.nb_scales,
                                                                                                      use_small_deblurgan, finest =False)
        self.generative_network_finest = GenerativeNetworkFiner(self.img_width, self.img_height, self.batch_size, self.nb_scales,
                                                                                                        use_small_deblurgan, finest = True)
        self.discriminative_network = DiscriminatorNetwork(self.img_width, self.img_height,
                                                           small_model=use_small_discriminator)
        self.vgg_network = VGGNetwork(self.img_width, self.img_height)

        ip = Input(shape=(3, self.img_width, self.img_height), name='x_generator')
                  
        aux_ip_finer = Input(shape=(3, self.img_width / 2, self.img_height / 2), name='aux_finer')
                  
        aux_ip_coarser = Input(shape=(3, self.img_width / 4, self.img_height / 4), name='aux_finer')
                  
        

        ip_gan = Input(shape=(3, self.img_width, self.img_height), name='x_discriminator') # Actual X images
        ip_vgg = Input(shape=(3, self.img_width, self.img_height), name='x_vgg') # Actual X images

        deblur_output = self.generative_network.create_deblur_model(ip)
        
        
        self.generative_model_ = Model(ip, deblur_output)
                  
        deblur_output_finer = self.generative_network_finer.create_deblur_model(ip, aux_ip_coarser)
                  
        self.generative_model_finer_ = Model(input = [ip, aux_ip_coarser], output =deblur_output_finer)
                  
        deblur_output_finer = self.generative_model_finer_([ip,self.generative_model_.output])
                  
        deblur_output_finest = self.generative_network_finest.create_deblur_model(ip, aux_ip_finer)
                  
        self.generative_model_finest_ = Model(input = [ip, aux_ip_finer], output =deblur_output_finest)
                  
        deblur_output_finest = self.generative_model_finest_([ip,deblur_output_finer])

        gan_output = self.discriminative_network.append_gan_network(ip_gan)
        self.discriminative_model_ = Model(ip_gan, gan_output)

        gan_output = self.discriminative_model_(deblur_output_finest)
        vgg_output = self.vgg_network.append_vgg_network(deblur_output_finest, ip_vgg)

        self.deblurgan_model_ = Model(input=[ip, ip_gan, ip_vgg], output=[gan_output, vgg_output])

        self.vgg_network.load_vgg_weight(self.deblurgan_model_)

        deblurgan_optimizer = Adam(lr = learning_rate)
        generator_optimizer = Adam(lr = learning_rate)
        discriminator_optimizer = Adam(lr = learning_rate)

        self.generative_model_.compile(generator_optimizer, dummy_loss)
        self.generative_model_finer_.compile(generator_optimizer, dummy_loss)
        self.generative_model_finest_.compile(generator_optimizer, dummy_loss)
        self.discriminative_model_.compile(discriminator_optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.deblurgan_model_.compile(deblurgan_optimizer, dummy_loss)

        return self.deblurgan_model_

    def train_full_model(self, image_dir, nb_images=50000, nb_epochs=450, use_small_deblurgan=False,
                         use_small_discriminator=False):

        self.build_deblurgan_model(use_small_deblurgan, use_small_discriminator)

        self._train_model(image_dir, nb_images, nb_epochs, load_generative_weights=True, load_discriminator_weights=True)

    def _train_model(self, image_dir, nb_images=80000, nb_epochs=450, pre_train_deblurgan=False,
                     pre_train_discriminator=False, load_generative_weights=False, load_discriminator_weights=False,
                     save_loss=True, disc_train_flip=0.1):


datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,
                             zoom_range=0.2,rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2)

        early_stop = False
        iteration = 0
        prev_improvement = -1

        if save_loss:
            loss_history = {'discriminator_loss' : [],
                            'discriminator_acc' : [],
                            'generator_loss' : [],
                            'val_psnr': [], }

        y_vgg_dummy = np.zeros((self.batch_size * 2, 3, self.img_width // 32, self.img_height // 32)) # 5 Max Pools = 2 ** 5 = 32

        print("Training DeblurGAN network")
        for i in range(nb_epochs):
            print()
            print("Epoch : %d" % (i + 1))
            if (i + 1) % 150 == 0:
                learning_rate = learning_rate / 10
            for x in datagen.flow_from_directory(image_dir, class_mode=None, batch_size=self.batch_size,
                                                 target_size=(self.img_width, self.img_height)):
                try:
                    t1 = time.time()

                    x_vgg = x.copy() * 255 # VGG input [0 - 255 scale]

                    # resize images
                    x_temp = x.copy()
                    x_temp = x_temp.transpose((0, 2, 3, 1))

                    x_generator = np.empty((self.batch_size, self.img_width, self.img_height, 3))
                    

                    for j in range(self.batch_size):
                        x_generator[j, :, :, :] = image.extract_patches_2d(x_temp[j], (64, 64))[0]

                    x_generator = x_generator.transpose((0, 3, 1, 2))

                    if iteration % 50 == 0 and iteration != 0 and not pre_train_discriminator:
                        print("Validation image..")
                        output_image_batch = self.generative_network.get_generator_output(x_generator,
                                                                                          self.deblurgan_model_)
                        if type(output_image_batch) == list:
                            output_image_batch = output_image_batch[0]

                        mean_axis = (0, 2, 3) if K.image_dim_ordering() == 'th' else (0, 1, 2)

                        average_psnr = 0.0

                        print('gen img mean :', np.mean(output_image_batch / 255., axis=mean_axis))
                        print('val img mean :', np.mean(x, axis=mean_axis))

                        for x_i in range(self.batch_size):
                            average_psnr += psnr(x[x_i], np.clip(output_image_batch[x_i], 0, 255) / 255.)

                        average_psnr /= self.batch_size

                        if save_loss:
                            loss_history['val_psnr'].append(average_psnr)

                        iteration += self.batch_size
                        t2 = time.time()

                        print("Time required : %0.2f. Average validation PSNR over %d samples = %0.2f" %
                              (t2 - t1, self.batch_size, average_psnr))

                        for x_i in range(self.batch_size):
                            real_path = "val_images/epoch_%d_iteration_%d_num_%d_real_.png" % (i + 1, iteration, x_i + 1)
                            generated_path = "val_images/epoch_%d_iteration_%d_num_%d_generated.png" % (i + 1,
                                                                                                        iteration,
                                                                                                        x_i + 1)

                            val_x = x[x_i].copy() * 255.
                            val_x = val_x.transpose((1, 2, 0))
                            val_x = np.clip(val_x, 0, 255).astype('uint8')

                            output_image = output_image_batch[x_i]
                            output_image = output_image.transpose((1, 2, 0))
                            output_image = np.clip(output_image, 0, 255).astype('uint8')

                            imsave(real_path, val_x)
                            imsave(generated_path, output_image)

                        '''
                        Don't train of validation images for now.

                        Note that if nb_epochs > 1, there is a chance that
                        validation images may be used for training purposes as well.

                        In that case, this isn't strictly a validation measure, instead of
                        just a check to see what the network has learned.
                        '''
                        continue

                    # Train only discriminator, disable training of deblurgan
                    self.discriminative_network.set_trainable(self.deblurgan_model_, value=True)
                    self.generative_network.set_trainable(self.deblurgan_model_, value=False)
                    self.generative_network_finer.set_trainable(self.deblurgan_model_, value=False)
                    self.generative_network_finest.set_trainable(self.deblurgan_model_, value=False)

                    X_pred = self.generative_model_.predict(x_generator, self.batch_size)
                    X_pred = self.generative_model_finer_.predict([x_generator,X_pred], self.batch_size)
                    X_pred = self.generative_model_finest_.predict([x_generator,X_pred], self.batch_size)

                    X = np.concatenate((X_pred, x * 255))

                    # Using soft and noisy labels
                    if np.random.uniform() > disc_train_flip:
                        # give correct classifications
                        y_gan = [0] * self.batch_size + [1] * self.batch_size
                    else:
                        # give wrong classifications (noisy labels)
                        y_gan = [1] * self.batch_size + [0] * self.batch_size

                    y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                    y_gan = to_categorical(y_gan, nb_classes=2)
                    y_gan = smooth_gan_labels(y_gan)

                    hist1 = self.discriminative_model_.fit(X, y_gan, verbose=0, batch_size=self.batch_size,
                                                          nb_epoch=1)

                    discriminator_loss = hist1.history['loss'][-1]

                    # Train only generator, disable training of discriminator
                    self.discriminative_network.set_trainable(self.deblurgan_model_, value=False)
                    self.generative_network.set_trainable(self.deblurgan_model_, value=True)
                    self.generative_network_finer.set_trainable(self.deblurgan_model_, value=True)
                    self.generative_network_finest.set_trainable(self.deblurgan_model_, value=True)

                    # Using soft labels
                    y_model = [1] * self.batch_size
                    y_model = np.asarray(y_model, dtype=np.int).reshape(-1, 1)
                    y_model = to_categorical(y_model, nb_classes=2)
                    y_model = smooth_gan_labels(y_model)

                    # Use custom bypass_fit to bypass the check for same input and output batch size
                    hist2 = bypass_fit(self.deblurgan_model_, [x_generator, x, x_vgg], [y_model, y_vgg_dummy],
                                       batch_size=self.batch_size, nb_epoch=1, verbose=0)

                    generative_loss = hist2.history['loss'][0]

                    if save_loss:
                        loss_history['discriminator_loss'].extend(hist1.history['loss'])
                        loss_history['discriminator_acc'].extend(hist1.history['acc'])
                        loss_history['generator_loss'].extend(hist2.history['loss'])

                    if prev_improvement == -1:
                        prev_improvement = discriminator_loss

                    improvement = (prev_improvement - discriminator_loss) / prev_improvement * 100
                    prev_improvement = discriminator_loss

                    iteration += self.batch_size
                    t2 = time.time()
                    print("Iter : %d / %d | Improvement : %0.2f percent | Time required : %0.2f seconds | "
                          "Discriminator Loss : %0.3f | Generative Loss : %0.3f" %
                          (iteration, nb_images, improvement, t2 - t1, discriminator_loss, generative_loss))

                    if iteration % 1000 == 0 and iteration != 0:
                        print("Saving model weights.")
                        self._save_loss_history(loss_history, pre_train_deblurgan, pre_train_discriminator, save_loss)

                    if iteration >= nb_images:
                        break

                except KeyboardInterrupt:
                    print("Keyboard interrupt detected. Stopping early.")
                    early_stop = True
                    break

            iteration = 0

            if early_stop:
                break

        print("Finished training DeblurGAN network. Saving history.")
        self._save_loss_history(loss_history, pre_train_deblurgan, pre_train_discriminator, save_loss)

    def _save_loss_history(self, loss_history, pre_train_deblurgan, pre_train_discriminator, save_loss):
        if save_loss:
            print("Saving loss history")

            with open('fulltrain losses.json', 'w') as f:
                json.dump(loss_history, f)

            print("Saved loss history")


if __name__ == "__main__":
    from keras.utils.visualize_util import plot

    # Path to MS COCO dataset
    coco_path = "/Users/f/Desktop/Super-Resolution-using-Generative-Adversarial-Networks-master/images/"


    deblurgan_network = DeblurGANNetwork(img_width=64, img_height=64, batch_size=4)
    deblurgan_network.build_deblurgan_model()
    plot(deblurgan_network.deblurgan_model_, 'DeblurGAN.png', show_shapes=True)

    # Fully train the DeblurGAN with VGG loss and Discriminator loss
    deblurgan_network.train_full_model(coco_path, nb_images=4000, nb_epochs=450)











