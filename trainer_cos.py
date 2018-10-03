import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始
import numpy as np
from tensorflow.contrib import layers
import config
from model import siamese
from model import siamese_loss,cos_loss
from tf_utils import get_one_batch_data
import datetime

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('gpu_id', 1, 'GPU id ')

tf.app.flags.DEFINE_string('train_dir', 'train', 'Directory to write checkpoints and logs')

tf.app.flags.DEFINE_string('checkpoint_path', 'ckpt', 'Initial weights path')

tf.app.flags.DEFINE_string('dataset_dir', '', 'training dataset directory')

tf.app.flags.DEFINE_integer('batch_size', config.batch_size, 'batch size')

tf.app.flags.DEFINE_integer('image_width', config.img_width, 'image width')

tf.app.flags.DEFINE_integer('image_height', config.img_height, 'image height')

tf.app.flags.DEFINE_integer('max_iter', 2000000, "max iterations")

tf.app.flags.DEFINE_float('base_lr', 0.001, 'init learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'learning rate decay steps')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'learning rate decay factor')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for rmsprop')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for optimizer')
tf.app.flags.DEFINE_bool('restore', False, 'restore ckpt')

FLAGS = tf.app.flags.FLAGS

def func_optimal(loss_val, var_list):
    with tf.variable_scope("optimizer"):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.base_lr, global_step,
                                                   decay_steps=FLAGS.decay_steps,
                                                   decay_rate=FLAGS.learning_rate_decay_factor,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_val, global_step)

        ##更新 BN
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(optimizer)
        update_op = tf.group(*update_ops)

    return update_op, global_step, learning_rate

def train():

    print("Setting up summary op...")
    summaries = set()
    fea_len = 1024
    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
    with tf.variable_scope('y') as scope:
        label = tf.placeholder(tf.float32, shape=[FLAGS.batch_size])

    print(1111111111111111111111)
    ##drop out
    with tf.name_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)


    with tf.variable_scope('siamese') as scope:

        ##左支
        out1 = siamese(x1, keep_prob, fea_len)

        ##参数共享
        scope.reuse_variables()

        ##右支
        out2 = siamese(x2, keep_prob, fea_len)

    print(2222222222222222222222)

    regularizer = layers.l2_regularizer(0.1)

    with tf.variable_scope('metrics') as scope:

        loss, dis = cos_loss(out1, out2, label)

        summaries.add(tf.summary.scalar("loss", loss))

        tf.add_to_collection('loss', loss)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        tf.add_to_collection('loss', regularization_loss)
        loss = tf.add_n(tf.get_collection('loss'))

    trainable_var = tf.trainable_variables()
    train_op, global_step, learning_rate = func_optimal(loss, trainable_var)

    # for var in tf.global_variables():
    #     print('&&&&&------------:', var.name)
    #     if 'is_training' not in var.name:
    #         summaries.add(tf.summary.histogram(var.name, var))

    #设置GPU
    sess_config = tf.ConfigProto(device_count={'GPU': FLAGS.gpu_id})
    sess_config.gpu_options.allow_growth = True

    ##样本graph
    img_1_batch, img_2_batch, label_batch = get_one_batch_data()

    with tf.Session(config=sess_config) as sess:

        print("Setting up Saver...")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        summary_writer = tf.summary.FileWriter("graph/siamese/", sess.graph)

        summary_op = tf.summary.merge(list(summaries))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

            if ckpt:
                print("Model restored...")
                saver.restore(sess, ckpt)

        print("training...")
        ##训练
        for itera in range(FLAGS.max_iter):


            img_1_train, img_2_train, label_train = sess.run([img_1_batch, img_2_batch, label_batch])

            if itera % 100 == 1:
                keep_prob_val = 1
            else:
                keep_prob_val = 0.8


            feed_dict_train = {x1: img_1_train, x2: img_2_train, label: label_train, keep_prob: keep_prob_val}

            _, train_loss, summary_str, dis_val, global_step_val, learning_rate_val = \
                sess.run([train_op, loss, summary_op, dis,
                          global_step, learning_rate], feed_dict=feed_dict_train)

            summary_writer.add_summary(summary_str, global_step_val)

            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
            if itera % 100 == 0:
                print('iter {}, time {}, train loss {},learn_rate {}'.format(itera,nowTime,
                                                                                     train_loss,
                                                                                     learning_rate_val))
                print(np.transpose(label_train))
                print(np.transpose(dis_val)[1])

            if itera % 1000 == 0:
                saver.save(sess, "ckpt/ckpt.ckpt" + str(global_step_val), global_step=1)

        summary_writer.close()


if __name__ == '__main__':
    train()
