from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.layers import Input
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Activation, Dropout, Embedding

import numpy as np
import pickle
import os

# base_path = 'D:/lyb/'
base_path = '/Users/mahaoyang/Downloads/'


def data2array(path):
    with open(path + 'DatasetA_train_20180813/label_list.txt', 'r') as f:
        label_list = dict()
        for line in f:
            line = line.strip('\n').split('\t')
            label_list[line[0]] = line[1]
            print(line)
        print(len(label_list))

    if not os.path.exists('train_list.pickle'):

        with open(path + 'DatasetA_train_20180813/train.txt', 'r') as f:
            train_list = dict()
            for line in f:
                line = line.strip('\n').split('\t')
                train_list[line[0]]['lable'] = line[1]

        for img in train_list:
            pic = load_img(path + 'DatasetA_train_20180813/train/' + img, target_size=(64, 64))
            pic = img_to_array(pic)
            pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
            train_list[img]['array'] = pic

        with open('train_list.pickle', 'wb') as f:
            pickle.dump(train_list, f)

    else:

        with open('train_list.pickle', 'rb') as f:
            train_list = pickle.load(f)

    with open(path + 'DatasetA_train_20180813/attributes_per_class.txt', 'r') as f:
        attributes_per_class = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            attributes_per_class[ii[0]] = ii[1:]

    with open(path + 'DatasetA_train_20180813/attribute_list.txt', 'r') as f:
        attribute_list = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            attribute_list[int(ii[0])] = ii[1:]

    return {'lable_list': label_list, 'train_list': train_list, 'attributes_per_class': attributes_per_class,
            'attribute_list': attribute_list}


if __name__ == '__main__':
    data2array(base_path)
