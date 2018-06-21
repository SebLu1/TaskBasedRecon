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
from Networks import UNet_multiple_classes
from Networks import UNet


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
        self.path = path_prefix+'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name, self.model_name, self.experiment_name)
        self.default_path = path_prefix+'Saves/{}/{}/{}/default_experiment/'.format(self.model.name, self.data_pip.name, self.model_name)
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

    def generate_raw_segmentation_data(self, batch_size, training_data=True):
        pics = np.zeros((batch_size, 512, 512, 1), dtype='float32')
        annos = np.zeros((batch_size, 512,512,1), dtype='float32')
        ul_nod = np.zeros(shape=(batch_size, 2))
        ul_rand = np.zeros(shape=(batch_size, 2))

        for i in range(batch_size):
            pic, vertices, nodules = self.data_pip.load_data(training_data=training_data)
            pics[i, ..., 0] = pic[...]
            annos[i,...,0] = nodules

            # find corresponding upper left corners for cut out
            x_cen, y_cen = self.data_pip.find_centre(vertices)
            j = 0
            upper_left = 0
            lower_right = 512
            while j < 100:
                centre_nod = [x_cen, y_cen] + np.random.randint(-20, 21, size=2)
                upper_left = centre_nod - 32
                lower_right = centre_nod + 32
                if upper_left[0] > 0 and upper_left[1] > 0 and lower_right[0] < 512 and lower_right[1] < 512:
                    j = 100
                j = j + 1
            ul_nod[i,:] = upper_left
            ul_rand[i,:]= np.random.randint(150, 314, size=2)
        return pics, annos, ul_nod, ul_rand

    def generate_reconstruction_data(self, batch_size, training_data = True, noise_level = None):
        if noise_level == None:
            noise_level = self.noise_level
        y = np.zeros((batch_size, self.measurement_space[0], self.measurement_space[1], 1), dtype='float32')
        x_true = np.zeros((batch_size, 512, 512, 1), dtype='float32')
        fbp = np.zeros((batch_size, 512, 512, 1), dtype='float32')

        for i in range(batch_size):
            pic, vertices, nodules = self.data_pip.load_data(training_data=training_data)
            data = self.model.forward_operator(pic)

            # add white Gaussian noise
            noisy_data = data + np.random.normal(size = self.measurement_space) * noise_level * np.average(np.abs(data))

            fbp [i, ..., 0] = self.model.inverse(noisy_data)
            x_true[i, ..., 0] = pic[...]
            y[i, ..., 0] = noisy_data
        return y, x_true, fbp

    def generate_training_data_mel(self, batch_size, training_data=True, noise_level = None):
        if noise_level == None:
            noise_level = self.noise_level
        y = np.zeros((batch_size, self.measurement_space[0], self.measurement_space[1]), dtype='float32')
        x_true = np.zeros((batch_size, 512, 512), dtype='float32')
        fbp = np.zeros((batch_size, 512, 512), dtype='float32')
        annos = np.zeros((batch_size, 512,512), dtype='float32')
        ul_nod = np.zeros(shape=(batch_size, 2))
        ul_rand = np.zeros(shape=(batch_size, 2))

        for i in range(batch_size):
            pic, vertices, nodules, mel = self.data_pip.load_data_mel(training_data=training_data)
            data = self.model.forward_operator(pic)

            # add white Gaussian noise
            noisy_data = data + np.random.normal(size = self.measurement_space) * noise_level * np.average(np.abs(data))

            fbp [i, ...] = self.model.inverse(noisy_data)
            x_true[i, ...] = pic[...]
            y[i, ...] = noisy_data
            annos[i,...] = nodules

            # find corresponding upper left corners for cut out
            x_cen, y_cen = self.data_pip.find_centre(vertices)
            j = 0
            upper_left = 0
            lower_right = 512
            while j < 100:
                centre_nod = [x_cen, y_cen] + np.random.randint(-20, 21, size=2)
                upper_left = centre_nod - 32
                lower_right = centre_nod + 32
                if upper_left[0] > 0 and upper_left[1] > 0 and lower_right[0] < 512 and lower_right[1] < 512:
                    j = 100
                j = j + 1
            ul_nod[i,:] = upper_left
            ul_rand[i,:]= np.random.randint(150, 314, size=2)

            # check if the random patch includes a nodule

        return y, x_true, fbp, annos, ul_nod, ul_rand, mel

    def generate_training_data(self, batch_size, training_data = True, noise_level= None, scaled = False):
        y, x_true, fbp, annos, ul_nod, ul_rand, mel = self.generate_training_data_mel(batch_size, training_data, noise_level)
        if scaled:
            annos = annos * mel
        return y, x_true, fbp, annos, ul_nod, ul_rand




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

    # # visualizes the quality of the current method
    # def visualize(self, true, fbp, guess, name):
    #     quality = np.average(np.sqrt(np.sum(np.square(true - guess), axis=(1, 2, 3))))
    #     print('Quality of reconstructed image: ' + str(quality) + 'SSIM: ' +
    #           str(ssim(true[-1,...,0], ut.cut_image(guess[-1,...,0]))))
    #     if self.colors == 1:
    #         t = true[-1,...,0]
    #         g = guess[-1, ...,0]
    #         p = fbp[-1, ...,0]
    #     else:
    #         t = true[-1,...]
    #         g = guess[-1, ...]
    #         p = fbp[-1, ...]
    #     plt.figure()
    #     plt.subplot(131)
    #     plt.imshow(ut.cut_image(t))
    #     plt.axis('off')
    #     plt.title('Original')
    #     plt.subplot(132)
    #     plt.imshow(ut.cut_image(p))
    #     plt.axis('off')
    #     plt.title('PseudoInverse')
    #     plt.suptitle('L2 :' + str(quality))
    #     plt.subplot(133)
    #     plt.imshow(ut.cut_image(g))
    #     plt.title('Reconstruction')
    #     plt.axis('off')
    #     plt.savefig(self.path + name + '.png')
    #     plt.close()

    def save(self, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        elif os.listdir(self.default_path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.default_path+'Data/'))
            print('Default Save restored')
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
    experiment_name = 'default_experiment'

    # learning rate for Adams
    learning_rate = 0.0005
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
                pics, annos = self.generate_segmentation_data(self.batch_size, training_data=False)
                iteration, loss = self.sess.run([self.global_step, self.loss], feed_dict={self.input_image: pics,
                                                     self.input_seg: annos})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(loss))

                # logging has to be adopted
                self.log(pics, annos)

        self.save(self.global_step)

    def evaluate(self):
        pass

class postprocessing(generic_framework):
    model_name = 'PostProcessing'

    # learning rate for Adams
    learning_rate = 0.0003
    # The batch size
    batch_size = 4

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def __init__(self):
        # call superclass init
        super(postprocessing, self).__init__()

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                dtype=tf.float32)
        # network output
        self.out = self.network.net(self.y)
        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))
        # optimizer
        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step)
        # logging tools
        tf.summary.scalar('Loss', self.loss)
        tf.summary.image('GroundTruth', self.true, max_outputs=1)
        tf.summary.image('FBP', self.y, max_outputs=1)
        tf.summary.image('Reconstruction', self.out, max_outputs=1)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, x, y):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.true: x,
                                                 self.y: y})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_reconstruction_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer, feed_dict={self.true: x_true,
                                                     self.y: fbp})
            if k % 20 == 0:
                iteration, loss = self.sess.run([self.global_step, self.loss], feed_dict={self.true: x_true,
                                                                                          self.y: fbp})
                print('Iteration: ' + str(iteration) + ', MSE: ' + str(loss))

                # logging has to be adopted
                self.log(x_true, fbp)

        self.save(self.global_step)

    def evaluate(self):
        y, x_true, fbp = self.generate_reconstruction_data(self.batch_size)

### bunch of methods that come in handy later
# method to slice in tensorflow
def extract_tensor(tensor, ul, batch_size, size = (64,64)):
    extract = tf.expand_dims(tensor[0,ul[0,0]:ul[0,0]+size[0],ul[0,1]:ul[0,1]+size[1],:], axis=0)
    for k in range(1, batch_size):
        new_slice = tf.expand_dims(tensor[k,ul[k,0]:ul[k,0]+size[0],ul[k,1]:ul[k,1]+size[1],:], axis=0)
        extract = tf.concat([extract, new_slice], axis=0)
    return extract

# compute weighted CE, overweighting 1st order mistakes by weight. Tensor 1 is recon, Tensor 2 ground truth segmentation
def CE(tensor1, tensor2, weight):
    # CE loss for 1st order mistakes
    segmentation_map1 = tf.multiply(tf.log(tensor1), tensor2)
    loss1 = - tf.reduce_mean(segmentation_map1)

    # CE loss for 2nd order mistakes
    segmentation_map2 = tf.multiply(tf.log(1 - tensor1), 1 - tensor2)
    loss2 = - tf.reduce_mean(segmentation_map2)

    # total loss
    return (weight * loss1 + loss2)

# method that computes the average and variance of a list of numbers
def mean_var(list):
    mean  = 0
    n = float(len(list)-1)
    if n == 0:
        n = 1.0

    for j in list:
        mean += j/n

    var = 0
    for j in list:
        var += ((j - mean)**2 )/n

    return mean, var

class joint_training(generic_framework):
    model_name = 'JointTraining_PostPr'
    experiment_name = 'default_experiment'

    # learning rate for Adams
    learning_rate = 0.0002
    # The batch size
    batch_size = 8
    # evaluation batch size
    eval_batch_size = 8
    # Convex weight alpha trading off between L2 and CE loss for joint reconstruction. 0 is pure L2, 1 is pure CE
    alpha = 0.7

    # methods to define the models used in framework
    def get_network_segmentation(self, size, colors):
        return UNet_segmentation(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def __init__(self):
        # call superclass init
        super(joint_training, self).__init__()
        self.segmenter = self.get_network_segmentation(self.image_size, self.colors)

        ### the reconstruction step
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.fbp = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                dtype=tf.float32)
        # network output
        with tf.variable_scope('Reconstruction'):
            self.out = self.network.net(self.fbp)
        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss_l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))

        ### the segmentation step
        self.segmentation = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                           dtype=tf.float32)
        self.ul_nod = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.ul_ran = tf.placeholder(shape=[None, 2], dtype=tf.int32)

        # slice the network output and segmentation using ul_nod and ul_ran
        self.pic_nod = extract_tensor(self.out, self.ul_nod, batch_size=self.batch_size)
        self.seg_nod = extract_tensor(self.segmentation, self.ul_nod, batch_size=self.batch_size)

        self.pic_ran = extract_tensor(self.out, self.ul_ran, batch_size=self.batch_size)
        self.seg_ran = extract_tensor(self.segmentation, self.ul_ran, batch_size=self.batch_size)

        with tf.variable_scope('Segmentation'):
            self.out_seg_nod = self.segmenter.net(self.pic_nod)
            self.out_seg_ran = self.segmenter.net(self.pic_ran)

        ce_nod = CE(self.out_seg_nod, self.seg_nod, weight=20)
        ce_ran = CE(self.out_seg_ran, self.seg_ran, weight=20)

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

        # collect summaries that are used for segmentation
        sum_seg = []

        tf.summary.scalar('Loss_L2', self.loss_l2)
        sum_seg.append(tf.summary.scalar('Loss_CE', self.ce))
        tf.summary.scalar('Loss_total', self.total_loss)
        with tf.name_scope('Reconstruction'):
            tf.summary.image('FBP', self.fbp, max_outputs=1)
            tf.summary.image('Original', self.true, max_outputs=1)
            tf.summary.image('Reconstruction', self.out, max_outputs=1)
        with tf.name_scope('Nodule_detection'):
            sum_seg.append(tf.summary.image('Nodule_pic', self.pic_nod, max_outputs=1))
            sum_seg.append(tf.summary.image('Nodule_seg', self.seg_nod, max_outputs=1))
            sum_seg.append(tf.summary.image('Nodule_out_seg', self.out_seg_nod, max_outputs=1))
        with tf.name_scope('Non_Nodule_detection'):
            sum_seg.append(tf.summary.image('Non-Nodule_pic', self.pic_ran, max_outputs=1))
            sum_seg.append(tf.summary.image('Non-Nodule_seg', self.seg_ran, max_outputs=1))
            sum_seg.append(tf.summary.image('Non-Nodule_out_seg', self.out_seg_ran, max_outputs=1))

        # set up the logger
        self.merged_seg_only = tf.summary.merge(sum_seg)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def pretrain_reconstruction(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer_recon, feed_dict={self.true: x_true,
                                                     self.fbp: fbp})
            if k % 20 == 0:
                y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size, training_data=False, noise_level=0.02)
                summary, iteration, loss = self.sess.run([self.merged,self.global_step, self.loss_l2],
                                                         feed_dict={self.true: x_true, self.y: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
                print('Iteration: ' + str(iteration) + ', MSE: ' + str(loss))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def pretrain_segmentation_true_input(self, steps):
        for k in range(steps):
            pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size= self.batch_size)
            self.sess.run(self.optimizer_seg, feed_dict={self.segmentation: annos, self.ul_nod:ul_nod,
                                                         self.ul_ran: ul_rand, self.out: pics})
            if k % 20 == 0:
                pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size=self.eval_batch_size, training_data=False)
                summary, iteration, loss = self.sess.run([self.merged_seg_only, self.global_step, self.ce],
                                                         feed_dict={self.segmentation: annos, self.ul_nod: ul_nod,
                                                                    self.ul_ran: ul_rand, self.out: pics})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(loss))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def pretrain_segmentation_reconstruction_input(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer_seg,feed_dict={self.true: x_true, self.fbp: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
            if k % 20 == 0:
                y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size, training_data=False,
                                                                                     noise_level=0.02)
                summary, iteration, ce = self.sess.run([self.merged,self.global_step, self.ce],
                                                         feed_dict={self.true: x_true, self.y: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(ce))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def joint_training(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer,feed_dict={self.true: x_true, self.fbp: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
            if k % 20 == 0:
                y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size, training_data=False,
                                                                                     noise_level=0.02)
                summary, iteration, ce = self.sess.run([self.merged,self.global_step, self.ce],
                                                         feed_dict={self.true: x_true, self.fbp: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(ce))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def compute(self, y, x_true, fbp, annos, ul_nod, ul_rand ):
        iteration, recon, nod, anno, seg = self.sess.run([self.global_step, self.out, self.pic_nod,
                                                                       self.seg_nod, self.out_seg_nod],
                                               feed_dict={self.true: x_true, self.fbp: fbp,
                                                          self.segmentation: annos,
                                                          self.ul_nod: ul_nod, self.ul_ran: ul_rand})
        return recon, nod, anno, seg

    # method to compute the average CE and L2 error and their variances
    def evaluate(self, test_size, direct_feed = False):
        ce_1 = []
        ce_2 = []
        ce_total = []
        l2 = []
        for k in range(int(test_size/16)):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size,
                                                                                 training_data=False,
                                                                                 noise_level=0.02)
            if not direct_feed:
                iteration, recon, nod, anno, seg = self.sess.run([self.global_step, self.out, self.pic_nod,
                                                                  self.seg_nod, self.out_seg_nod],
                                                                 feed_dict={self.true: x_true, self.fbp: fbp,
                                                                            self.segmentation: annos,
                                                                            self.ul_nod: ul_nod, self.ul_ran: ul_rand})
            else:
                iteration, recon, nod, anno, seg = self.sess.run([self.global_step, self.out, self.pic_nod,
                                                                  self.seg_nod, self.out_seg_nod],
                                                                 feed_dict={self.out: x_true,
                                                                            self.segmentation: annos,
                                                                            self.ul_nod: ul_nod, self.ul_ran: ul_rand})
            for j in range(self.eval_batch_size):
                ### compute ce_1, ce_2 and ce_total
                # CE loss for 1st order mistakes
                segmentation_map1 = np.multiply(np.log(seg[j,...,0]), anno[j,...,0])
                loss1 = - np.average(segmentation_map1)
                ce_1.append(loss1)

                # CE loss for 2nd order mistakes
                segmentation_map2 = np.multiply(np.log(1 - seg[j,...,0]), 1 - anno[j,...,0])
                loss2 = - np.average(segmentation_map2)
                ce_2.append(loss2)

                # total loss
                ce_total.append((20 * loss1 + loss2))

                ### compute the l2 loss
                mistake = np.square(recon[j,...,0] - x_true[j,...,0])
                l2.append(np.sqrt(np.sum(mistake)))
        return mean_var(ce_1), mean_var(ce_2), mean_var(ce_total), mean_var(l2)

class joint_training_mal(generic_framework):
    model_name = 'Postprocessing_Mal'
    experiment_name = 'default_experiment'

    # channels in the segmentation
    channels = 6
    # learning rate for Adams
    learning_rate = 0.000025
    # The batch size
    batch_size = 2
    # Convex weight alpha trading off between L2 and CE loss for joint reconstruction. 0 is pure L2, 1 is pure CE
    alpha = 0.7

    # methods to define the models used in framework
    def get_network_segmentation(self, channels):
        return UNet_multiple_classes(channels=channels)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def vis_seg(self, seg):
        weights = tf.constant([[0]])
        for k in range(1,self.channels):
            ad = tf.constant([[k]])
            weights = tf.concat([weights, ad], axis=0)
        return tf.tensordot(seg, weights, axes=[[-1],[0]])

    def segment(self, pic, ohl, name):
        with tf.variable_scope('Segmentation'):
            out_seg= self.segmenter.net(pic)

        weight_non_nod = tf.constant([[0.05]])
        class_weighting = tf.concat([weight_non_nod], tf.ones(shape=[self.channels, 1]), axis = 0)
        location_weight = tf.tensordot(ohl, class_weighting, axes=[[3], [0]])

        raw_ce = tf.nn.softmax_cross_entropy_with_logits(ohl, out_seg)
        weighted_ce = tf.multiply(raw_ce, location_weight)
        ce = tf.reduce_mean(weighted_ce)

        # visualization of segmentation
        seg = self.vis_seg(tf.nn.softmax(ohl, axis=-1))
        seg_net = self.vis_seg(out_seg)

        # the tensorboard logging
        with tf.name_scope(name):
            self.sum_seg.append(tf.summary.image('Image', pic, max_outputs=1))
            self.sum_seg.append(tf.summary.image('Annotation', seg, max_outputs=1))
            self.sum_seg.append(tf.summary.image('Segmentation',seg_net, max_outputs=1))

        return ce


    def __init__(self):
        # call superclass init
        super(joint_training_mal, self).__init__()
        self.network = self.get_network(size = self.image_size, colors = self.colors)
        self.segmenter = self.get_network_segmentation(self.channels)

        # logging list
        self.sum_seg = []

        ### the reconstruction step
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1]],
                                   dtype=tf.float32)
        self.fbp = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1]],
                                dtype=tf.float32)
        # network output
        with tf.variable_scope('Reconstruction'):
            self.out = self.network.net(tf.expand_dims(self.fbp, axis = 3))
        # compute loss
        data_mismatch = tf.square(self.out - tf.expand_dims(self.true, axis = 3))
        self.loss_l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))

        ### the segmentation step
        self.segmentation = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1]],
                                           dtype=tf.int32)
        self.seg_ohl = tf.one_hot(self.segmentation, depth = self.channels)
        self.ul_nod = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.ul_ran = tf.placeholder(shape=[None, 2], dtype=tf.int32)

        # segmentation of patch containing nodule
        pic_nod = extract_tensor(self.out, self.ul_nod, batch_size=self.batch_size)
        seg_nod = extract_tensor(self.seg_ohl, self.ul_nod, batch_size=self.batch_size)
        ce_nod = self.segment(pic_nod, seg_nod, name='Nodule')

        # segmentation of random patch
        pic_ran = extract_tensor(self.out, self.ul_ran, batch_size=self.batch_size)
        seg_ran = extract_tensor(self.seg_ohl, self.ul_ran, batch_size=self.batch_size)
        ce_ran = self.segment(pic_ran, seg_ran, name='Random')

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
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def pretrain_reconstruction(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer_recon, feed_dict={self.true: x_true,
                                                     self.fbp: fbp})
            if k % 20 == 0:
                y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size, training_data=False, noise_level=0.02)
                summary, iteration, loss = self.sess.run([self.merged,self.global_step, self.loss_l2],
                                                         feed_dict={self.true: x_true, self.y: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
                print('Iteration: ' + str(iteration) + ', MSE: ' + str(loss))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def pretrain_segmentation_true_input(self, steps):
        for k in range(steps):
            pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size= self.batch_size)
            self.sess.run(self.optimizer_seg, feed_dict={self.segmentation: annos, self.ul_nod:ul_nod,
                                                         self.ul_ran: ul_rand, self.out: pics})
            if k % 20 == 0:
                pics, annos, ul_nod, ul_rand = self.generate_raw_segmentation_data(batch_size=self.batch_size, training_data=False)
                summary, iteration, loss = self.sess.run([self.merged_seg_only, self.global_step, self.ce],
                                                         feed_dict={self.segmentation: annos, self.ul_nod: ul_nod,
                                                                    self.ul_ran: ul_rand, self.out: pics})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(loss))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def pretrain_segmentation_reconstruction_input(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer_seg,feed_dict={self.true: x_true, self.fbp: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
            if k % 20 == 0:
                y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size, training_data=False,
                                                                                     noise_level=0.02)
                summary, iteration, ce = self.sess.run([self.merged,self.global_step, self.ce],
                                                         feed_dict={self.true: x_true, self.y: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(ce))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def joint_training(self, steps):
        for k in range(steps):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.batch_size, noise_level=0.02)
            self.sess.run(self.optimizer,feed_dict={self.true: x_true, self.fbp: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
            if k % 20 == 0:
                y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size, training_data=False,
                                                                                     noise_level=0.02)
                summary, iteration, ce = self.sess.run([self.merged,self.global_step, self.ce],
                                                         feed_dict={self.true: x_true, self.fbp: fbp,
                                                                    self.segmentation: annos,
                                                                    self.ul_nod:ul_nod, self.ul_ran: ul_rand})
                print('Iteration: ' + str(iteration) + ', CE: ' + str(ce))

                # logging has to be adopted
                self.writer.add_summary(summary, iteration)
        self.save(self.global_step)

    def compute(self, y, x_true, fbp, annos, ul_nod, ul_rand ):
        iteration, recon, nod, anno, seg = self.sess.run([self.global_step, self.out, self.pic_nod,
                                                                       self.seg_nod, self.out_seg_nod],
                                               feed_dict={self.true: x_true, self.fbp: fbp,
                                                          self.segmentation: annos,
                                                          self.ul_nod: ul_nod, self.ul_ran: ul_rand})
        return recon, nod, anno, seg

    # method to compute the average CE and L2 error and their variances
    def evaluate(self, test_size, direct_feed = False):
        ce_1 = []
        ce_2 = []
        ce_total = []
        l2 = []
        for k in range(int(test_size/16)):
            y, x_true, fbp, annos, ul_nod, ul_rand = self.generate_training_data(self.eval_batch_size,
                                                                                 training_data=False,
                                                                                 noise_level=0.02)
            if not direct_feed:
                iteration, recon, nod, anno, seg = self.sess.run([self.global_step, self.out, self.pic_nod,
                                                                  self.seg_nod, self.out_seg_nod],
                                                                 feed_dict={self.true: x_true, self.fbp: fbp,
                                                                            self.segmentation: annos,
                                                                            self.ul_nod: ul_nod, self.ul_ran: ul_rand})
            else:
                iteration, recon, nod, anno, seg = self.sess.run([self.global_step, self.out, self.pic_nod,
                                                                  self.seg_nod, self.out_seg_nod],
                                                                 feed_dict={self.out: x_true,
                                                                            self.segmentation: annos,
                                                                            self.ul_nod: ul_nod, self.ul_ran: ul_rand})
            for j in range(self.batch_size):
                ### compute ce_1, ce_2 and ce_total
                # CE loss for 1st order mistakes
                segmentation_map1 = np.multiply(np.log(seg[j,...,0]), anno[j,...,0])
                loss1 = - np.average(segmentation_map1)
                ce_1.append(loss1)

                # CE loss for 2nd order mistakes
                segmentation_map2 = np.multiply(np.log(1 - seg[j,...,0]), 1 - anno[j,...,0])
                loss2 = - np.average(segmentation_map2)
                ce_2.append(loss2)

                # total loss
                ce_total.append((20 * loss1 + loss2))

                ### compute the l2 loss
                mistake = np.square(recon[j,...,0] - x_true[j,...,0])
                l2.append(np.sqrt(np.sum(mistake)))
        return mean_var(ce_1), mean_var(ce_2), mean_var(ce_total), mean_var(l2)


