# -*- coding:utf-8 -*-
import tensorflow as tf
slim = tf.contrib.slim

class cnn_model(object):

    def __init__(self, inputs, size, is_training):
        self.input = inputs
        self.size = size

        batch_norm_params = {
            'is_training': is_training,
            'center': True,
            'scale': True,
            'decay': 0.9997,
            'epsilon': 0.001,
        }
        # weights_init = tf.truncated_normal_initializer(stddev=0.02)
        # regularizer = tf.contrib.layers.l2_regularizer(0.9)

        with tf.variable_scope('simpleCNN'):
            with slim.arg_scope([tf.contrib.layers.batch_norm], **batch_norm_params):
                self.endpoints = self.build_model()


    def build_model(self):
        ##搭建网络结构

        endpoints = {}

        ##224*224*3

        endpoints['input'] = self.input

        net = tf.contrib.layers.conv2d(self.input, num_outputs=32 * self.size,  # 生成的滤波器的数量
                                        activation_fn=None,
                                        weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                        kernel_size=(3, 3), stride=(2, 2), padding="SAME")

        endpoints['conv1'] = net

        net = tf.contrib.layers.batch_norm(net)
        endpoints['bn1'] = net

        net = tf.nn.relu6(net)
        endpoints['relu1'] = net

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        endpoints['pool1'] = net


        ##56*56*(8*size)
        net = tf.contrib.layers.conv2d(net, num_outputs=32 * self.size,  # 生成的滤波器的数量
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                       kernel_size=(3, 3), stride=(2, 2), padding="SAME")
        endpoints['conv2'] = net

        net = tf.contrib.layers.batch_norm(net)
        endpoints['bn2'] = net

        net = tf.nn.relu6(net)
        endpoints['relu2'] = net

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        endpoints['pool2'] = net


        ##28*28*(16*size)
        net = tf.contrib.layers.conv2d(net, num_outputs=64 * self.size,  # 生成的滤波器的数量
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                       kernel_size=(3, 3), stride=(2, 2), padding="SAME")

        endpoints['conv3'] = net

        net = tf.contrib.layers.batch_norm(net)
        endpoints['bn3'] = net

        net = tf.nn.relu6(net)
        endpoints['relu3'] = net

        ##14*14*(32*size)
        net = tf.contrib.layers.conv2d(net, num_outputs=128*self.size,  # 生成的滤波器的数量
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                       kernel_size=(3, 3), stride=(2, 2), padding="SAME")


        endpoints['conv4'] = net

        net = tf.contrib.layers.batch_norm(net)
        endpoints['bn4'] = net

        net = tf.nn.relu6(net)
        endpoints['relu4'] = net

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        endpoints['pool4'] = net

        ##7*7*(64*size)
        net = tf.contrib.layers.conv2d(net, num_outputs=128 * self.size,  # 生成的滤波器的数量
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                       kernel_size=(3, 3), stride=(1, 1), padding="SAME")
        endpoints['conv5'] = net

        ##7*7*(64*size)
        net = tf.contrib.layers.conv2d(net, num_outputs=256 * self.size,  # 生成的滤波器的数量
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer,  # 设置weight的值是正态分布的随机值
                                       kernel_size=(2, 2), stride=(1, 1), padding="VALID")
        endpoints['conv6'] = net


        net = tf.reshape(net, [net.get_shape()[0].value, -1])

        W = tf.Variable(tf.zeros([net.get_shape()[-1].value, 512]))  # w为784*10的矩阵，输入层为784个像素点
        b = tf.Variable(tf.zeros([1024]))  # 输出为10，偏离值也为10

        net = tf.matmul(net, W) + b
        endpoints['fc'] = net

        return endpoints