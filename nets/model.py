import tensorflow as tf

from nets import cnn_model
from nets import SENet_inception

def cosine(q,a):
    #pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    #pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    #pooled_mul_12 = tf.reduce_sum(q * a, 1)
    #score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
 
    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(q), axis=1))
    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=1))
    #内积
    x3_x4 = tf.reduce_sum(tf.multiply(q, a), axis=1)
    cosin = x3_x4 / (x3_norm * x4_norm)
    score = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm)+1e-8, name="scores")

    return score

def cos_loss(out1,out2,y):

    dis = cosine(out1, out2)
    loss = tf.reduce_mean(tf.square(dis - y))

    return loss, dis


def siamese_loss(out1,out2,y):

    diff = out1 - out2 #tf.sqrt(tf.reduce_sum(tf.square(out1 - out2), axis=1))

    sim_w = tf.Variable(tf.truncated_normal(shape=[diff.get_shape()[-1].value, 2],
                                            stddev=0.05, mean=0), name='sim_w')
    sim_b = tf.Variable(tf.zeros(2), name='sim_b')
    #print(sim_w, sim_b, diff)
    pred = tf.add(tf.matmul(diff, sim_w), sim_b)

    logist = tf.nn.softmax(pred)

    print(logist)
    print(y,tf.argmax(logist, 1))
    correct_prediction = tf.equal(tf.cast(tf.argmax(logist, 1), tf.float32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y, tf.int32), logits=pred)
    
    loss = tf.reduce_mean(loss)
    #loss = tf.reduce_mean(tf.square(pred - y))
 
    #loss = tf.cond(loss > 5.0, lambda: tf.sqrt(loss), lambda: loss)

    #return loss, pred, loss, loss, loss
    return loss, logist, accuracy, sim_w, sim_b

def siamese(inputs, keep_prob, fea_len):

    #cnnModel = cnn_model.cnn_model(inputs, 1, True)
    cnnModel    = SENet_inception.SENetResNeXt(inputs, 1, True)

    bn_fc2 = cnnModel.endpoints["fc"]


    with tf.name_scope('fc3') as scope:
        w_fc3 = tf.Variable(tf.truncated_normal(shape=[bn_fc2.get_shape()[-1].value, fea_len],
                                                stddev=0.05, mean=0), name='w_fc3')
        b_fc3 = tf.Variable(tf.zeros(fea_len), name='b_fc3')
        fc3 = tf.add(tf.matmul(bn_fc2, w_fc3), b_fc3)

    #fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob, noise_shape=None, seed=None, name=None)

    return fc3
