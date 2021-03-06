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
from Networks import fully_convolutional
from Networks import UNet_multiple_classes

# This class provides methods necessary
class generic_framework(object):
    model_name = 'no_model'

    # set the noise level used for experiments
    noise_level = 0.02

    # channels in the segmentation
    channels = 6

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)


    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

        self.data_pip = self.get_Data_pip()
        self.colors = 1
        self.image_size = (512,512)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()
        self.network = self.get_network(size=self.image_size, colors=self.colors)


        # finding the correct path for saving models
        name = platform.node()
        path_prefix = ''
        if name == 'LAPTOP-E6AJ1CPF':
            path_prefix=''
        elif name == 'motel':
            path_prefix='/local/scratch/public/sl767/TaskBasedRecon/'
        self.path = path_prefix+'Saves/Noise_Level_{}/Channels_{}/{}/{}/'.format(self.noise_level, self.channels, self.model_name, self.experiment_name)
        self.default_path = path_prefix+'Saves/Noise_Level_{}/Channels_{}/{}/default_experiment/'.format(self.noise_level, self.channels, self.model_name)
        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # generate needed folder structure
        self.generate_folders()


    def generate_raw_segmentation_data(self, batch_size, training_data=True, scaled=True, from_source=False):
        pics = np.zeros((batch_size, 512, 512,1), dtype='float32')
        annos = np.zeros((batch_size, 512,512), dtype='float32')
        ul_nod = np.zeros(shape=(batch_size, 2))
        ul_rand = np.zeros(shape=(batch_size, 2))

        for i in range(batch_size):
            if from_source:
                pic, nodules, ul_n, ul_r, mel = self.data_pip.load_from_source(id=((i % 8)+1))
            else:
                pic, nodules, ul_n, ul_r, mel = self.data_pip.load_data(training_data=training_data)
            pics[i, ...,0] = pic
            if scaled:
                nodules = nodules * mel
            annos[i,...] = nodules
            ul_nod[i,:] = ul_n
            ul_rand[i,:]= ul_r
        return pics, annos, ul_nod, ul_rand

    def generate_training_data(self, batch_size, training_data=True, noise_level=None, scaled=True, from_source=False):
        if noise_level is None:
            noise_level = self.noise_level
        y = np.zeros((batch_size, self.measurement_space[0], self.measurement_space[1],1), dtype='float32')
        x_true = np.zeros((batch_size, 512, 512,1), dtype='float32')
        fbp = np.zeros((batch_size, 512, 512,1), dtype='float32')
        annos = np.zeros((batch_size, 512,512), dtype='float32')
        ul_nod = np.zeros(shape=(batch_size, 2), dtype='float32')
        ul_rand = np.zeros(shape=(batch_size, 2), dtype='float32')

        for i in range(batch_size):
            if from_source:
                pic, nodules, ul_n, ul_r, mel = self.data_pip.load_from_source(id=((i % 8)+1))
            else:
                pic, nodules, ul_n, ul_r, mel = self.data_pip.load_data(training_data=training_data)
            data = self.model.forward_operator(pic)

            # add white Gaussian noise
            noisy_data = data + np.random.normal(size = self.measurement_space) * noise_level * np.average(np.abs(data))

            fbp[i, ...,0] = self.model.inverse(noisy_data)
            x_true[i, ...,0] = pic
            y[i, ...,0] = noisy_data
            if scaled:
                nodules = nodules * mel
            annos[i,...] = nodules
            ul_nod[i,:] = ul_n
            ul_rand[i,:]= ul_r

        return y, x_true, fbp, annos, ul_nod, ul_rand


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

    def save(self, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        elif os.path.exists(self.default_path+'Data/'):
                if os.listdir(self.default_path+'Data/'):
                    saver.restore(self.sess, tf.train.latest_checkpoint(self.default_path+'Data/'))
                    print('Default Save restored')
                else:
                    print('No save found')
        else:
            print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    ### generic method for subclasses
    def deploy(self, true, guess, measurement):
        pass

# joint postprocessing and segmentation
class postprocessing(generic_framework):
    model_name = 'Postprocessing'
    experiment_name = 'default_experiment'

    # channels in the segmentation
    channels = 6
    # scaled variable setting format of input training data
    scaled = True
    # The batch size
    batch_size = 4
    # fix the noise level
    noise_level = 0.02

    # some static methods that can come in handy
    @staticmethod
    def extract_tensor(tensor, ul, size=(64, 64)):
        size_tf = tf.constant(size)
        offset_np = [0,0]
        for k in range(2):
            offset_np[k] = float(size[k]/2.0)
        offset = tf.constant(offset_np)
        cut = tf.image.extract_glimpse(tensor, size_tf, ul+offset, normalized=False, centered=False)
        return cut

    # methods to define the models used in framework
    def get_network_segmentation(self, channels):
        return UNet_multiple_classes(channels=channels)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def reconstruction_model(self, y, fbp):
        return self.network.net(fbp)

    def vis_seg(self, seg):
        seg_fl = tf.cast(seg, tf.float32)
        weights = tf.constant([[0]], dtype=tf.float32)
        for k in range(1, self.channels):
            ad = tf.constant([[k]], dtype=tf.float32)
            weights = tf.concat([weights, ad], axis=0)
        return tf.tensordot(seg_fl, weights, axes=[[-1],[0]])

    def segment(self, pic, ohl, name):
        with tf.variable_scope('Segmentation'):
            out_seg= self.segmenter.net(pic)

        weight_non_nod = tf.constant([[0.05]])
        class_weighting = tf.concat([weight_non_nod, tf.ones(shape=[self.channels-1, 1])], axis = 0)
        location_weight = tf.tensordot(ohl, class_weighting, axes=[[3], [0]])

        raw_ce = tf.nn.softmax_cross_entropy_with_logits(labels=ohl, logits=out_seg)
        weighted_ce = tf.multiply(tf.expand_dims(raw_ce, axis=3), location_weight)
        ce = tf.reduce_mean(weighted_ce)

        # visualization of segmentation
        seg = self.vis_seg(ohl)
        seg_net = self.vis_seg(ut.softmax(out_seg))

        # the tensorboard logging
        with tf.name_scope(name):
            self.sum_seg.append(tf.summary.image('Image', pic, max_outputs=2))
            self.sum_seg.append(tf.summary.image('Annotation', seg, max_outputs=2))
            self.sum_seg.append(tf.summary.image('Segmentation',seg_net, max_outputs=2))
            self.sum_seg.append(tf.summary.image('Weight_map', location_weight, max_outputs=2))
        return ce


    def __init__(self, experiment_name, c, learning_rate):
        # call superclass init
        super(postprocessing, self).__init__(experiment_name)
        # Convex weight alpha trading off between L2 and CE loss for joint reconstruction. 0 is pure L2, 1 is pure CE
        self.alpha = c
        # learning rate for Adams
        self.learning_rate = learning_rate

        self.network = self.get_network(size = self.image_size, colors = self.colors)
        self.segmenter = self.get_network_segmentation(self.channels)

        # logging list
        self.sum_seg = []

        ### the reconstruction step
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1],1],
                                   dtype=tf.float32)
        self.fbp = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1],1],
                                dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.measurement_space[0], self.measurement_space[1], 1],
                                dtype=tf.float32)

        # network output
        with tf.variable_scope('Reconstruction'):
            self.out = self.reconstruction_model(self.y, self.fbp)
        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss_l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))

        ### the segmentation step
        self.segmentation = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1]],
                                           dtype=tf.int32)
        self.seg_ohl = tf.one_hot(self.segmentation, depth = self.channels)
        self.ul_nod = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.ul_ran = tf.placeholder(shape=[None, 2], dtype=tf.float32)

        # segmentation of patch containing nodule
        self.pic_nod = self.extract_tensor(self.out, self.ul_nod)
        self.seg_nod = self.extract_tensor(self.seg_ohl, self.ul_nod)
        ce_nod = self.segment(self.pic_nod, self.seg_nod, name='Nodule')

        # segmentation of random patch
        self.pic_ran = self.extract_tensor(self.out, self.ul_ran)
        self.seg_ran = self.extract_tensor(self.seg_ohl, self.ul_ran)
        ce_ran = self.segment(self.pic_ran, self.seg_ran, name='Random')

        self.ce = ce_nod+ce_ran

        # total loss for joint training. Weight of 50 is to align different scales of ce and lossL2
        self.total_loss = self.alpha * self.ce * 50 + (1-self.alpha) * self.loss_l2

        ### optimizers
        # Train Reconstruction Only
        rec_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Reconstruction')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer_recon = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_l2,
                                                                             global_step=self.global_step,
                                                                             var_list=rec_var)
        # Train Segmentation Only
        seg_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Segmentation')
        self.optimizer_seg = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ce,
                                                                             global_step=self.global_step,
                                                                             var_list=seg_var)
        # Train everything
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                             global_step=self.global_step)

        ### logging tools
        tf.summary.scalar('Loss_L2', self.loss_l2)
        self.sum_seg.append(tf.summary.scalar('Loss_CE', self.ce))
        tf.summary.scalar('Loss_total', self.total_loss)
        with tf.name_scope('Reconstruction'):
            tf.summary.image('FBP', self.fbp, max_outputs=1)
            tf.summary.image('Original', self.true, max_outputs=1)
            tf.summary.image('Reconstruction', self.out, max_outputs=1)

        # set up the logger
        self.merged_seg_only = tf.summary.merge(self.sum_seg)
        self.merged = tf.summary.merge_all()
        self.writer_random = tf.summary.FileWriter(self.path + 'Logs/random/',
                                            self.sess.graph)
        self.writer_static = tf.summary.FileWriter(self.path + 'Logs/fixed/',
                                            self.sess.graph)
        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, direct_feed = False):
        if direct_feed:
            pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size=self.batch_size,
                                                                               scaled=self.scaled, from_source=True)
            summary, iteration, loss = self.sess.run([self.merged_seg_only, self.global_step, self.ce],
                                                     feed_dict={self.segmentation: annos, self.ul_nod: ul_nod,
                                                                self.ul_ran: ul_rand, self.out: pics})
            self.writer_static.add_summary(summary, iteration)
            print('Iteration: ' + str(iteration) + ', CE: ' + str(loss))
            pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size=self.batch_size,
                                                                               training_data= False,
                                                                               scaled=self.scaled, from_source=False)
            summary, iteration = self.sess.run([self.merged_seg_only, self.global_step],
                                                     feed_dict={self.segmentation: annos, self.ul_nod: ul_nod,
                                                                self.ul_ran: ul_rand, self.out: pics})
            self.writer_random.add_summary(summary, iteration)
        else:
            # evaluate on the same samples all the time
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=self.noise_level,
                                                                                 scaled=self.scaled, from_source=True)
            summary, iteration, ce, mse = self.sess.run([self.merged, self.global_step, self.ce, self.loss_l2],
                                                   feed_dict={self.true: x_true, self.fbp: fbp, self.y: y,
                                                              self.segmentation: annos,
                                                              self.ul_nod: ul_nod, self.ul_ran: ul_rand})
            self.writer_static.add_summary(summary, iteration)
            print('Iteration: ' + str(iteration) + ', CE: ' + str(ce) + ', MSE: ' +str(mse))

            # evaluate on random samples
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, training_data=False,
                                                                                 noise_level=self.noise_level,
                                                                                 scaled=self.scaled, from_source=False)
            summary, iteration= self.sess.run([self.merged, self.global_step],
                                                   feed_dict={self.true: x_true, self.fbp: fbp, self.y: y,
                                                              self.segmentation: annos,
                                                              self.ul_nod: ul_nod, self.ul_ran: ul_rand})
            self.writer_random.add_summary(summary, iteration)

    def pretrain_reconstruction(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=self.noise_level, scaled=self.scaled)
            self.sess.run(self.optimizer_recon, feed_dict={self.true: x_true,
                                                     self.fbp: fbp, self.y: y})
            if k % 20 == 0:
                self.log()
        self.save(self.global_step)

    def pretrain_segmentation_true_input(self, steps):
        for k in range(steps):
            pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size= self.batch_size, scaled=self.scaled)
            self.sess.run(self.optimizer_seg, feed_dict={self.segmentation: annos, self.ul_nod:ul_nod,
                                                         self.ul_ran: ul_rand, self.out: pics})
            if k % 20 == 0:
                self.log(direct_feed=True)
        self.save(self.global_step)


    def pretrain_segmentation_reconstruction_input(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=self.noise_level, scaled=self.scaled)
            self.sess.run(self.optimizer_seg,feed_dict={self.true: x_true, self.fbp: fbp, self.y: y,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
            if k % 20 == 0:
                self.log()
        self.save(self.global_step)

    def joint_training(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=self.noise_level, scaled=self.scaled)
            self.sess.run(self.optimizer,feed_dict={self.true: x_true, self.fbp: fbp, self.y: y,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
            if k % 20 == 0:
                self.log()
        self.save(self.global_step)

    def evaluate(self, y, x_true, fbp, annos, ul_nod, ul_rand):
        ce, mse, total = self.sess.run([self.ce, self.loss_l2, self.total_loss],
                                               feed_dict={self.true: x_true, self.fbp: fbp, self.y:y,
                                                          self.segmentation: annos,
                                                          self.ul_nod: ul_nod, self.ul_ran: ul_rand})
        return ce, mse, total

# changes reconstruction algorithm to iterative descent
class iterative_gradient_desc(postprocessing):
    model_name = 'LearnedGD'
    recursions = 4

    # take a leightweight network for a single iteration
    def get_network(self, size, colors):
        fully_convolutional(size=size, colors=colors)

    # the recursion model
    def reconstruction_model(self, y, fbp):
        x = fbp
        for k in range(self.recursions):
            # get gradient of data term
            grad = self.model.tensorflow_adjoint_operator(self.model.tensorflow_operator(x) - y)
            # network with gradient of data term and current guess as input
            with tf.variable_scope('Iteration_' + str(k)):
                x = self.network.net(tf.concat((grad, x), axis=3))
        return x

