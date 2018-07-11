import tensorflow as tf
from util import lrelu
import util as ut


def bin_softmax(x):
    e = tf.exp(x)
    o = tf.ones(shape=tf.shape(x))
    return tf.divide(e, o+e)

def downsampling_block(tensor, name, filters, reuse, kernel = [5,5]):
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv1', reuse=reuse, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv2', reuse=reuse, activation=tf.nn.relu)
        pool = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        return pool

def upsampling_block(tensor, name, filters, reuse, kernel = [5,5]):
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv1', reuse=reuse, activation=tf.nn.relu)
        upsample = tf.layers.conv2d_transpose(inputs=conv1, filters=filters, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv1',
                                           reuse=reuse, activation=tf.nn.relu)
        return upsample

class small_UNet(object):
    def __init__(self, size, colors, parameter_sharing = True):
        self.colors = colors
        self.size = size
        self.parameter_sharing = parameter_sharing
        self.used = False

    def raw_net(self, input, reuse):
        # same shape conv
        pre1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5],
                                      padding="same", name='pre1', reuse=reuse, activation=tf.nn.relu)
        # downsampling 1
        down1 = downsampling_block(tensor= pre1, name='down1', filters= 32, reuse=reuse)

        # downsampling 2
        down2 = downsampling_block(tensor=down1, name='down2', filters=64, reuse=reuse)

        # upsampling 3
        up3 = upsampling_block(tensor=down2, name='up3', filters=64, reuse=reuse)
        con3 = tf.concat([up3, down1], axis = 3)

        # upsampling 4
        up4 = upsampling_block(tensor=con3, name='up4', filters=32, reuse=reuse)
        con4 = tf.concat([up4, pre1], axis = 3)


        post1 = tf.layers.conv2d(inputs=con4, filters=16, kernel_size=[5, 5],
                                  padding="same", name='post1',
                                  reuse=reuse,  activation=tf.nn.relu)

        post2 = tf.layers.conv2d(inputs=post1, filters=1, kernel_size=[5, 5],
                                  padding="same", name='post2',
                                  reuse=reuse, activation = tf.nn.relu)
        return post2

    def net(self, input):
        output = self.raw_net(input, reuse=self.used)
        if self.parameter_sharing:
            self.used = True
        return output

class UNet_multiple_classes(object):

    def __init__(self, channels,  parameter_sharing=True):
        self.channels = channels
        self.parameter_sharing = parameter_sharing
        self.used = False

    def raw_net(self, input, reuse):
        # same shape conv
        pre1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5],
                                padding="same", name='pre1', reuse=reuse, activation=tf.nn.relu)
        # downsampling 1
        down1 = downsampling_block(tensor=pre1, name='down1', filters=32, reuse=reuse)

        # downsampling 2
        down2 = downsampling_block(tensor=down1, name='down2', filters=64, reuse=reuse)

        # downsampling 3
        down3 = downsampling_block(tensor=down2, name='down3', filters=64, reuse=reuse)

        # downsampling 4
        down4 = downsampling_block(tensor=down3, name='down4', filters=64, reuse=reuse)

        # upsampling 1
        up1 = upsampling_block(tensor=down4, name='up1', filters=64, reuse=reuse)
        con1 = tf.concat([up1, down3], axis=3)

        # upsampling 2
        up2 = upsampling_block(tensor=con1, name='up2', filters=64, reuse=reuse)
        con2 = tf.concat([up2, down2], axis=3)

        # upsampling 3
        up3 = upsampling_block(tensor=con2, name='up3', filters=64, reuse=reuse)
        con3 = tf.concat([up3, down1], axis=3)

        # upsampling 4
        up4 = upsampling_block(tensor=con3, name='up4', filters=32, reuse=reuse)
        con4 = tf.concat([up4, pre1], axis=3)

        post1 = tf.layers.conv2d(inputs=con4, filters=16, kernel_size=[5, 5],
                                 padding="same", name='post1',
                                 reuse=reuse, activation=tf.nn.relu)
        post2 = tf.layers.conv2d(inputs=post1, filters=self.channels, kernel_size=[5, 5],
                                 padding="same", name='post2',
                                 reuse=reuse)

        return post2

    def net(self, input):
        output = self.raw_net(input, reuse=self.used)
        if self.parameter_sharing:
            self.used = True
        return output


class fully_convolutional(small_UNet):

    def raw_net(self, input, reuse):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=8, kernel_size=[5, 5],
                                 padding="same", name='conv1', reuse=reuse, activation=tf.nn.relu)
        # 64
        conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[5, 5],
                                 padding="same", name='conv2', reuse=reuse, activation=tf.nn.relu)
        # 32
        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3, 3],
                                 padding="same", name='conv3', reuse=reuse, activation=tf.nn.relu)
        # 64
        conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=[5, 5],
                                 padding="same", name='conv4',reuse=reuse, activation=tf.nn.relu)
        # 128
        conv5 = tf.layers.conv2d(inputs=conv4, filters=8, kernel_size=[5, 5],
                                 padding="same", name='conv5', reuse=reuse, activation=tf.nn.relu)
        output = tf.layers.conv2d(inputs=conv5, filters=self.colors, kernel_size=[5, 5],
                                  padding="same", name='conv6', reuse=reuse)
        return output

