import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
import numpy as np
slim = tf.contrib.slim

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, scope):
    return tf.contrib.layers.batch_norm(x,  activation_fn=tf.nn.relu, scope=scope)

def Relu(x):
    return tf.nn.relu6(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Fully_connected(x, units=2, layer_name='fully_connected'):
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


class SENetResNeXt():
    def __init__(self, args):

        self.input = args.input
        self.size = args.size
        batch_norm_params = {
            'is_training': args.is_training,
            'center': True,
            'scale': True,
            'decay': 0.9997,
            'epsilon': 0.001,
        }

        with tf.variable_scope('SENet_ResNeXt'):
            with slim.arg_scope([tf.contrib.layers.batch_norm], **batch_norm_params):
                self.endpoints = self._build_model()


    def first_layer(self, x, out_channel, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=out_channel, kernel=[3, 3], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, scope=scope+'_batch1')
            x = Relu(x)
            return x


    def transform_layer(self, x, depth, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def split_layer(self, input_x, depth, cardinality, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, depth, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):

            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, stride, out_dim, layer_name,
                       depth = 32, cardinality=8, reduction_ratio=4):

        input_dim = int(np.shape(input_x)[-1])

        x = self.split_layer(input_x, depth, cardinality=cardinality,stride=stride, layer_name='split_layer_'+layer_name+'_')

        x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_name)

        x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio,
                                              layer_name='squeeze_layer_'+layer_name)

        pad_channel = int((out_dim - input_dim) / 2)

        if stride == 2:
            pad_input_x = Average_pooling(input_x)
        else:
            pad_input_x = input_x

        pad_input_x = tf.pad(pad_input_x,[[0, 0], [0, 0], [0, 0], [pad_channel, pad_channel]])  # [?, height, width, channel]

        input_x = Relu(x + pad_input_x)

        return input_x


    def _build_model(self):

        endpoints = {}

        net_stride = [2, 2]
        net_out_dim = [64, 64]

        endpoints['input'] = self.input

        net = self.first_layer(self.input, 32*self.size, 2, scope='first_layer')
        endpoints['conv1'] = net

        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu6(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        net = self.residual_layer(net, stride=2, out_dim=int(32*self.size), layer_name="0")
        endpoints['conv2'] = net

        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu6(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        for i in range(len(net_stride)):
            net = self.residual_layer(net, stride=net_stride[i], out_dim=int(net_out_dim[i]*self.size), layer_name=str(i+1))
            endpoints["conv"+str(i+3)] = net

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        endpoints['pool4'] = net

        ##4*4*(64*size)
        net = tf.contrib.layers.conv2d(net, num_outputs=int(128*self.size),  # 生成的滤波器的数量
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                       kernel_size=(3, 3), stride=(1, 1), padding="SAME")
        endpoints['conv5'] = net

        return endpoints
