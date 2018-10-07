import tensorflow as tf

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始
import numpy as np
from tensorflow.contrib import layers
import config
from nets.model import *
import datetime
from tf_utils import *

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('gpu_id', 1, 'GPU id ')

tf.app.flags.DEFINE_string('train_dir', 'train', 'Directory to write checkpoints and logs')

tf.app.flags.DEFINE_string('checkpoint_path', 'ckpt-all', 'Initial weights path')

tf.app.flags.DEFINE_string('dataset_dir', '', 'training dataset directory')

tf.app.flags.DEFINE_integer('batch_size', config.batch_size, 'batch size')

tf.app.flags.DEFINE_integer('image_width', config.img_width, 'image width')

tf.app.flags.DEFINE_integer('image_height', config.img_height, 'image height')

tf.app.flags.DEFINE_integer('max_iter', 2000000, "max iterations")

tf.app.flags.DEFINE_float('base_lr', 0.001, 'init learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'learning rate decay steps')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.99, 'learning rate decay factor')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.99, 'Decay term for rmsprop')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.99, 'Momentum')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for optimizer')
tf.app.flags.DEFINE_bool('restore', True, 'restore ckpt')

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
    fea_len = 1000

    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
    with tf.variable_scope('y') as scope:
        label = tf.placeholder(tf.float32, shape=[FLAGS.batch_size])
        label_a = tf.placeholder(tf.float32, shape=[FLAGS.batch_size])
        label_b = tf.placeholder(tf.float32, shape=[FLAGS.batch_size])

    print(1111111111111111111111)
    ##drop out
    with tf.name_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)


    with tf.variable_scope('siamese') as scope:

        ##左支
        out1 = siamese(x1, keep_prob, fea_len)

        w = tf.Variable(tf.truncated_normal(shape=[out1.get_shape()[-1].value, 2000],
                                                stddev=0.05, mean=0), name='w')
        b = tf.Variable(tf.zeros(2000), name='b')
        # print(sim_w, sim_b, diff)
        predA = tf.add(tf.matmul(out1, w), b)

        correct_prediction = tf.equal(tf.cast(tf.argmax(predA, 1), tf.float32), label_a)
        accuracyA = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        lossA = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(label_a, tf.int32), logits=predA)
        lossA = tf.reduce_mean(lossA)

        ##参数共享
        scope.reuse_variables()

        ##右支
        out2 = siamese(x2, keep_prob, fea_len)

        predB = tf.add(tf.matmul(out2, w), b)

        correct_prediction = tf.equal(tf.cast(tf.argmax(predB, 1), tf.float32), label_b)
        accuracyB = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        lossB = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(label_b, tf.int32), logits=predB)
        lossB = tf.reduce_mean(lossB)


    print(2222222222222222222222)

    regularizer = layers.l2_regularizer(0.1)

    with tf.variable_scope('metrics') as scope:

        lossS, pred, accuracy, sim_w, sim_b = siamese_loss(out1, out2, label)

        summaries.add(tf.summary.scalar("loss", lossS))

        loss = lossA + lossB + lossS

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
    #device_count = {'GPU': FLAGS.gpu_id}
    sess_config = tf.ConfigProto()
    #sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allow_growth = True

    ##样本graph
    #img_1_batch, img_2_batch, label_batch = get_one_batch_data()

    with tf.Session(config=sess_config) as sess:

        print("Setting up Saver...")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        #
        # summary_writer = tf.summary.FileWriter("graph/siamese/", sess.graph)
        #
        # summary_op = tf.summary.merge(list(summaries))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

            if ckpt:
                print("Model restored...")
                print(ckpt)
                saver.restore(sess, ckpt)

        print("training...")
        ##训练


        du = data_utils()


        for itera in range(FLAGS.max_iter):

            img_1_train, img_2_train, label_train, _, _, labelA_train, labelB_train= du.get_one_batch(FLAGS.batch_size)
            #img_1_train, img_2_train, label_train = sess.run([img_1_batch, img_2_batch, label_batch])

            if itera % 100 == 1:
                keep_prob_val = 1
            else:
                keep_prob_val = 0.8


            feed_dict_train = {x1: img_1_train, x2: img_2_train,
                               label: label_train,label_a:labelA_train, label_b:labelB_train,
                               keep_prob: keep_prob_val}

            _, lossS_val,lossA_val,lossB_val, pred_val,acc_val,accA_val,accB_val, global_step_val, learning_rate_val = \
                sess.run([train_op, lossS,lossA,lossB, pred, accuracy,accuracyA,accuracyB,
                          global_step, learning_rate], feed_dict=feed_dict_train)

            #summary_writer.add_summary(summary_str, global_step_val)

            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
            if itera % 100 == 0:
                print('iter {}, time {}, lossS_val {}, lossA_val {}, lossB_val {}, acc_val {}, accA {}, accB {}, learn_rate {}'.format(itera,
                                                                                     nowTime,
                                                                                     lossS_val,lossA_val,lossB_val,
                                                                                     acc_val,accA_val,accB_val,
                                                                                     learning_rate_val))
                print("label",np.transpose(label_train))
                print("pred",np.transpose(pred_val)[-1])
                print("label_a",np.transpose(labelA_train))
                print("label_b", np.transpose(labelB_train))

            if itera % 1000 == 0:
                saver.save(sess, "ckpt-all/ckpt.ckpt" + str(global_step_val), global_step=1)

        #summary_writer.close()

def inference():

    print("Setting up summary op...")
    summaries = set()
    FLAGS.batch_size = 1
    fea_len = 1024

    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])

    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])

    with tf.variable_scope('siamese') as scope:
        ##左支
        out1 = siamese(x1, 1, fea_len)
        ##参数共享
        scope.reuse_variables()
        ##右支
        out2 = siamese(x2, 1, fea_len)

    with tf.variable_scope('metrics') as scope:

        diff = out1 - out2

        sim_w = tf.Variable(tf.truncated_normal(shape=[diff.get_shape()[-1].value, 2],
                                                stddev=0.05, mean=0), name='sim_w')
        sim_b = tf.Variable(tf.zeros(2), name='sim_b')
        print(sim_w, sim_b, diff)
        pred = tf.add(tf.matmul(diff, sim_w), sim_b)

        logist = tf.nn.softmax(pred)

        print(logist)

    print(x1)
    print(out1)

    #设置GPU
    sess_config = tf.ConfigProto(device_count={'GPU': FLAGS.gpu_id})
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        print("Setting up Saver...")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

        if ckpt:
            print("Model restored...")
            print(ckpt)
            saver.restore(sess, ckpt)
        else:
            print(None)
            return

        output_graph_def = tf.graph_util. \
            convert_variables_to_constants(sess,
                                           sess.graph.as_graph_def(),
                                           ['siamese/fc3/Add'])

        with tf.gfile.FastGFile('model/output_graph.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    train()
