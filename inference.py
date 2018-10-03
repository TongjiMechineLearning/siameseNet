#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-19 下午12:46
# @File    : demo.py
# @Software: PyCharm
# @Author  : wxw
# @Contact : xwwei@lighten.ai
# @Desc    :

import tensorflow as tf
import cv2
import numpy as np
from scipy.spatial.distance import pdist
class SiameseNet(object):

    ##默认输入4*4
    def __init__(self, img_shape=[224,224]):

        self.img_w = 224
        self.img_h = 224

        self.photo_w = 0
        self.photo_h = 0

        self.pb_path = "model/output_graph.pb"

        self.output = "siamese/fc3/Add:0"
        self.input = 'input_x1/Placeholder:0'

        with tf.gfile.FastGFile(self.pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        self.session = tf.Session()
        self.fea_val = self.session.graph.get_tensor_by_name(self.output)


    def inference(self, img_data):

        self.photo_w = img_data.shape[1]
        self.photo_h = img_data.shape[0]
        img_data = self.get_input_data(img_data)

        fea_vec = self.session.run(self.fea_val,
                           {self.input: img_data})


        print(fea_vec.shape)
        return fea_vec



    def get_input_data(self, img_data):

        img_resize = cv2.resize(img_data, (self.img_w, self.img_h))
        valid_img = img_resize.reshape([1, self.img_h, self.img_w, 3])
        valid_img = (np.asarray(valid_img, np.float32) / 255 - 0.5)

        for i in range(3):
            for j in range(3):
                print(i, j, valid_img[0, i, j, ...])

        return valid_img





if __name__ == '__main__':

    img_path = "image.jpg"
    siameseNet = SiameseNet()
    img = cv2.imread(img_path)


    img_path = "test.jpg"
    #img_sub = img[0:int(img.shape[0]*9/10), ...]

    img_sub = cv2.imread(img_path)

    #data = np.zeros(img.shape)

    fea_vec = siameseNet.inference(img)
    fea_vec_sub = siameseNet.inference(img_sub)


    print(fea_vec)
    print(fea_vec_sub)

    dist2 = pdist(np.vstack([fea_vec, fea_vec_sub]), 'cosine')
    dist1 = np.linalg.norm(fea_vec - fea_vec_sub)

    print(dist1, dist2)




