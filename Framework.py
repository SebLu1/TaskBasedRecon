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
from Networks import fully_convolutional
from Networks import UNet_segmentation


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

    # method to generate training data for the segmentation algorithm
    # method to generate training data given the current model type
    def generate_segmentation_data(self, batch_size, training_data=True):
        pics = np.zeros(shape=[batch_size * 2, 64, 64, 1])
        annos = np.zeros(shape=[batch_size * 2, 64, 64, 1])
        for k in range(batch_size):
            pic, vertices, nodules = self.data_pip.load_data(training_data=training_data)
            pic_cut, nod_cut, pic_rand, nod_rand = self.data_pip.cut_data(pic, nodules, vertices)

            pics[k, ..., 0] = pic_cut
            annos[k, ..., 0] = nod_cut

            pics[k + batch_size, ..., 0] = pic_rand
            annos[k + batch_size, ..., 0] = nod_rand
        return pics, annos


    # # method to generate training data given the current model type
    # def generate_training_data(self, batch_size, training_data = True):
    #     y = np.empty((batch_size, self.measurement_space[0], self.measurement_space[1], self.colors), dtype='float32')
    #     x_true = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
    #     fbp = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
    #     nodules = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
    #     for i in range(batch_size):
    #         pic, nod, _ = self.data_pip.load_nodule(training_data=training_data)
    #         data = self.model.forward_operator(pic)
    #
    #         # add white Gaussian noise
    #         noisy_data = data + np.random.normal(size = self.measurement_space) * self.noise_level
    #
    #         fbp [i, ..., 0] = self.model.inverse(noisy_data)
    #         x_true[i, ..., 0] = pic[...]
    #         y[i, ..., 0] = noisy_data
    #         nodules[i,...,0] = nod
    #     return y, x_true, fbp, nodules

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
    model_name = 'PureSegmentation'

    # learning rate for Adams
    learning_rate = 0.001
    # The batch size
    batch_size = 16

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return UNet_segmentation(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def __init__(self):
        # call superclass init
        super(pure_segmentation, self).__init__()

        # placeholder
        self.input_image = tf.placeholder(shape=(None, 64, 64, 1), dtype=tf.float32)
        self.input_seg = tf.placeholder(shape=(None, 64, 64, 1), dtype=tf.float32)

        output_seg = self.network.net(self.input_image)

        # CE loss for 1st order mistakes
        segmentation_map1 = tf.multiply(tf.log(output_seg), self.input_seg)
        loss1 = - tf.reduce_mean(segmentation_map1)

        # CE loss for 2nd order mistakes
        segmentation_map2 = tf.multiply(tf.log(1 - output_seg), 1 - self.input_seg)
        loss2 = - tf.reduce_mean(segmentation_map2)

        # total loss
        self.loss = 20 * loss1 + loss2

        # optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step)
        # logging tools
        tf.summary.scalar('Loss_overall', self.loss)
        tf.summary.scalar('Loss_no_module', loss1)
        tf.summary.scalar('Loss_abundant_module', loss2)
        tf.summary.image('Scan', self.input_image)
        tf.summary.image('Annotation', self.input_seg)
        tf.summary.image('Segmentation', output_seg)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, pics, annos):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.input_image: pics,
                                                self.input_seg: annos})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            pics, annos = self.generate_segmentation_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.input_image: pics,
                                                     self.input_seg: annos})
            if k % 20 == 0:
                iteration, loss = self.sess.run([self.global_step, self.loss], feed_dict={self.input_image: pics,
                                                     self.input_seg: annos})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(loss))

                # logging has to be adopted
                self.log(pics, annos)

        self.save(self.global_step)

    def evaluate(self):
        pass