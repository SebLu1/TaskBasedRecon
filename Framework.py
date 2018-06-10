import random
import numpy as np
import scipy.ndimage
import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
import odl
import odl.contrib.tensorflow
from skimage.measure import compare_ssim as ssim

from scipy.misc import imresize
import tensorflow as tf
import util as ut

from forward_models import ct
from DataProcessing import LUNA
from Networks import binary_classifier
from Networks import UNet
from Networks import fully_convolutional


# This class provides methods necessary
class generic_framework(object):
    model_name = 'no_model'
    experiment_name = 'default_experiment'

    # set the noise level used for experiments
    noise_level = 0.02

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return binary_classifier(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)


    def __init__(self):
        self.data_pip = self.get_Data_pip()
        self.colors = 1
        self.image_size = (512,512)
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()


        # finding the correct path for saving models
        name = platform.node()
        path_prefix = ''
        if name == 'LAPTOP-E6AJ1CPF':
            path_prefix=''
        elif name == 'motel':
            path_prefix='/local/scratch/public/sl767/DeepAdversarialRegulariser/'
        self.path = path_prefix+'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name, self.model_name, self.experiment_name)
        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # generate needed folder structure
        self.generate_folders()

    # method to generate training data given the current model type
    def generate_training_data(self, batch_size, training_data = True):
        y = np.empty((batch_size, self.measurement_space[0], self.measurement_space[1], self.colors), dtype='float32')
        x_true = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        fbp = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        nodules = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        for i in range(batch_size):
            pic, nod, _ = self.data_pip.load_nodule(training_data=training_data)
            data = self.model.forward_operator(pic)

            # add white Gaussian noise
            noisy_data = data + np.random.normal(size = self.measurement_space) * self.noise_level

            fbp [i, ..., 0] = self.model.inverse(noisy_data)
            x_true[i, ..., 0] = pic[...]
            y[i, ..., 0] = noisy_data
            nodules[i,...,0] = nod
        return y, x_true, fbp, nodules

    # puts in place the folders needed to save the results obtained with the current model
    def generate_folders(self):
        paths = {}
        paths['Image Folder'] = self.path + 'Images'
        paths['Saves Folder'] = self.path + 'Data'
        paths['Logging Folder'] = self.path + 'Logs'
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    # visualizes the quality of the current method
    def visualize(self, true, fbp, guess, name):
        quality = np.average(np.sqrt(np.sum(np.square(true - guess), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality) + 'SSIM: ' +
              str(ssim(true[-1,...,0], ut.cut_image(guess[-1,...,0]))))
        if self.colors == 1:
            t = true[-1,...,0]
            g = guess[-1, ...,0]
            p = fbp[-1, ...,0]
        else:
            t = true[-1,...]
            g = guess[-1, ...]
            p = fbp[-1, ...]
        plt.figure()
        plt.subplot(131)
        plt.imshow(ut.cut_image(t))
        plt.axis('off')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(ut.cut_image(p))
        plt.axis('off')
        plt.title('PseudoInverse')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(133)
        plt.imshow(ut.cut_image(g))
        plt.title('Reconstruction')
        plt.axis('off')
        plt.savefig(self.path + name + '.png')
        plt.close()

    def save(self, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        else:
            print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    ### generic method for subclasses
    def deploy(self, true, guess, measurement):
        pass

class pure_segmentation(generic_framework):
    pass