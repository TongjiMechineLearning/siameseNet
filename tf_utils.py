# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import config
from multiprocessing import Pool
from data_manager import *
import random
import _thread
import copy

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


##数据打包
def write_all_data(data_count, list_name, start_index, end_index, th_idx):

    tf_one_count = 1000

    ##打包数据用于手指检测与定位
    for index in range(start_index, end_index):
        try:
            print(th_idx, data_count, index)

            if (index-start_index) % tf_one_count == 0:
                file_name = config.tf_data_path + "/{}_{}.tfrecoder".\
                    format(th_idx, int(index / tf_one_count))

                writer = tf.python_io.TFRecordWriter(file_name)

            img_path = list_name[index]

            img_1 = cv2.imread(img_path)
            img_2 = Image_Data_Generator(img_1)

            if img_1 == [] or img_2 == [] :
                continue

            img_1 = cv2.resize(img_1, (config.img_width, config.img_height))
            img_2 = cv2.resize(img_2, (config.img_width, config.img_height))

            if config.is_show:
                cv2.imshow("img1", img_1)
                cv2.imshow("img2", img_2)
                cv2.waitKey(0)

            img_1 = img_1.astype(np.uint8)
            img_2 = img_2.astype(np.uint8)
            label = 1
            write_one_data(writer, img_1, img_2, label)

            for i in range(config.neg_number):
                ##
                if np.random.randint(10) > 5:
                    img_1 = Image_Data_Generator(img_1)
                else:
                    img_1 = img_1

                ##
                idx = np.random.randint(data_count)
                if idx == index and index > 0:
                    idx = index - 1
                else:
                    idx = index + 1

                img_path = list_name[idx]
                img_2 = cv2.imread(img_path)

                if np.random.randint(10) > 5:
                    img_2 = Image_Data_Generator(img_2)
                else:
                    img_2 = img_2

                if img_1 == [] or img_2 == [] :
                    continue

                img_1 = cv2.resize(img_1, (config.img_width, config.img_height))
                img_2 = cv2.resize(img_2, (config.img_width, config.img_height))

                if config.is_show:
                    cv2.imshow("img1", img_1)
                    cv2.imshow("img2", img_2)
                    cv2.waitKey(0)

                img_1 = img_1.astype(np.uint8)
                img_2 = img_2.astype(np.uint8)
                label = 0
                write_one_data(writer, img_1, img_2, label)
        except:
            print("error data!")
    writer.close()

def write_one_data(writer, img_1, img_2, label):


    example = tf.train.Example(features=tf.train.Features(
        feature={
            'img_1': _bytes_feature(img_1.tobytes()),
            'img_2': _bytes_feature(img_2.tobytes()),
            'label': _int64_feature(label)
        }))

    #print("success!")

    serialized = example.SerializeToString()
    writer.write(serialized)


def get_one_batch_data():

    file_name = config.tr_data_path + "*.tfrecoder"

    file_list = tf.gfile.Glob(file_name)

    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )
    _, serialized_example = reader.read(filename_queue)

    batch = tf.train.batch([serialized_example], config.batch_size, capacity=config.batch_size)

    features = tf.parse_example(batch, features={
        'img_1': tf.FixedLenFeature([], tf.string),
        "img_2": tf.FixedLenFeature([], tf.string),
        "label": tf.VarLenFeature(tf.int64)
    })

    img_1 = features["img_1"]
    img_2 = features["img_2"]
    label = features["label"]

    img_1_batch = tf.decode_raw(img_1, tf.uint8)

    img_2_batch = tf.decode_raw(img_2, tf.uint8)

    label = tf.sparse_tensor_to_dense(label)

    img_1_batch = tf.cast(tf.reshape(img_1_batch,
                                      [config.batch_size, config.img_height, config.img_width,
                                       config.img_channel]), tf.float32)

    img_2_batch = tf.cast(tf.reshape(img_2_batch,
                                      [config.batch_size, config.img_height, config.img_width,
                                       config.img_channel]), tf.float32)

    return img_1_batch / 255.0 - 0.5, img_2_batch / 255.0 - 0.5, tf.reshape(label, [-1])


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

from  data_manager import  *

class data_utils:
    def __init__(self):
        folder_data = config.img_data_path
        ##遍历全部文件
        self.file_list = []
        for path in folder_data:
            self.file_list = listdir(path, self.file_list)
        self.idx_neg = 0
        self.idx_pos = 0

        #print(self.file_list)
        self.list_a = [i for i in self.file_list]
        self.list_b = [i for i in self.file_list]
        self.list_c = [i for i in self.file_list]

        random.shuffle(self.list_a)
        random.shuffle(self.list_b)
        random.shuffle(self.list_c)

        # print(self.list_a[0])
        # print(self.list_b[0])
        # print(self.list_c[0])


    def get_one_batch(self, batch_size):

        pos_num = batch_size // 3

        img_1_batch = np.zeros([batch_size, 224, 224, 3], np.float32)
        img_2_batch = np.zeros([batch_size, 224, 224, 3], np.float32)
        label_batch = np.zeros([batch_size], np.float32)
        path_a = [i for i in range(batch_size)]
        path_b = [i for i in range(batch_size)]
        idx = [i for i in range(0, batch_size)]

        random.shuffle(idx)
        #print(idx, self.idx_neg)
        count = 0

        while(count < pos_num):

            path = self.list_a[self.idx_pos]
            img_data = cv2.imread(path)
            if img_data is not None:
                img_1 = random_crop_image(img_data)
                angle = np.random.randint(-45, 45)
                img_2 = rotate_image(img_data, angle, img_data.shape[1], img_data.shape[0])
                img_3 = cv2.resize(img_data, (224, 224))

                img_1_batch[idx[count]] = cv2.resize(img_1, (224, 224))
                if random.randint(0, 10) >= 5:
                    img_2_batch[idx[count], ...] = cv2.resize(img_2, (224, 224))
                else:
                    img_2_batch[idx[count], ...] = cv2.resize(img_3, (224, 224))

                label_batch[idx[count]] = 0
                path_a[idx[count]] = path
                path_b[idx[count]] = path
                count += 1

            self.idx_pos += 1
            if self.idx_pos >= len(self.list_a):
                random.shuffle(self.list_a)
                self.idx_pos = 0


        while (count < batch_size):
            self.idx_neg += 1

            path_1 = self.list_b[len(self.list_b) - self.idx_neg - 1]
            path_2 = self.list_c[len(self.list_c) - self.idx_neg - 1]

            if path_1 != path_2:

                img_data_1 = cv2.imread(path_1)
                img_data_2 = cv2.imread(path_2)

                if img_data_1 is not None and img_data_2 is not None:

                    if random.randint(0,10) >= 5:
                        if random.randint(0,10)>=5:
                            img_data_1 = random_crop_image(img_data_1)
                        else:
                            angle = np.random.randint(0, 45)
                            img_data_1 = rotate_image(img_data_1, angle, img_data_1.shape[1], img_data_1.shape[0])

                    img_data_1 = cv2.resize(img_data_1, (224, 224))

                    if random.randint(0,10) >= 5:
                        if random.randint(0,10)>=5:
                            img_data_2 = random_crop_image(img_data_2)
                        else:
                            angle = np.random.randint(0, 45)
                            img_data_2 = rotate_image(img_data_2, angle, img_data_2.shape[1], img_data_2.shape[0])

                    img_data_2 = cv2.resize(img_data_2, (224, 224))


                    img_1_batch[idx[count], ...] = cv2.resize(img_data_1, (224, 224))

                    img_2_batch[idx[count], ...] = cv2.resize(img_data_2, (224, 224))

                    label_batch[idx[count], ...] = 1

                    path_a[idx[count]] = path_1
                    path_b[idx[count]] = path_2

                    count += 1

            self.idx_neg += 1
            if self.idx_neg >= len(self.list_a):
                random.shuffle(self.list_b)
                random.shuffle(self.list_c)
                self.idx_neg = 0

        return img_1_batch, img_2_batch, label_batch, path_a, path_b

if __name__ == '__main__':


    data_utils = data_utils()


    img_1_batch, img_2_batch, label_batch, path_a, path_b = data_utils.get_one_batch(64)

    for i in range(64):

        cv2.imshow("a",np.reshape(img_1_batch[i, ...], [224,224,3])/ 225.0)
        cv2.imshow("b",np.reshape(img_2_batch[i, ...], [224,224,3])/ 255.0)

        print(label_batch[i])
        print(path_a[i])
        print(path_b[i])
        cv2.waitKey(0)

    # list_name = Get_All_Data()
    # data_count = list_name.__len__()
    # thread_num = 1
    # thread_count = data_count // thread_num
    #
    # try:
    #     for i in range(thread_num):
    #         start_index = i * thread_count
    #         end_index   = (i + 1) * thread_count
    #         if thread_num - 1 == i:
    #             end_index = data_count - 1
    #         _thread.start_new_thread(write_all_data, (data_count, list_name, start_index, end_index, i))
    #
    # except:
    #     print("Error: 无法启动线程")
    #
    # while 1:
    #     pass
