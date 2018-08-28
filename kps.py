from keras.layers import Embedding

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.utils import to_categorical

from keras.preprocessing.image import load_img, img_to_array

import os
from PIL import Image
from scipy import misc
import numpy as np
import random
import cv2

x = []
y = dict()
img_size = (200, 200)

with open('D:/lyb/DatasetA_train_20180813/label_list.txt', 'r') as f:
# with open('/Users/mahaoyang/Downloads/DatasetA_train_20180813/label_list.txt', 'r') as f:
    label_list = []
    for line in f:
        line = line.strip('\n').split('\t')
        label_list.append(line[0])
        print(line)
    print(len(label_list))

with open('D:/lyb/DatasetA_train_20180813/train.txt', 'r') as f:
# with open('/Users/mahaoyang/Downloads/DatasetA_train_20180813/train.txt', 'r') as f:
    for line in f:
        line = line.strip('\n').split('\t')
        x.append(line[0])
        y[line[0]] = line[1]
print(len(x), len(y))
# directory = '/Users/mahaoyang/Downloads/DatasetA_train_20180813/train/'
directory = 'D:/lyb/DatasetA_train_20180813/train/'
# for imgname in x:  # 参数是文件夹路径 directory
#
#     # print(imgname)
#
#     # img = Image.open(directory + imgname[0])
#     # arr = np.asarray(img, dtype=np.float32)  # 数组维度(128, 192, 3)
#     # # print(img.size, arr.shape)
#     # arr = image.img_to_array(img)  # 数组维度(128, 192, 3)
#     # # print(img.size, arr.shape)
#     imgname[0] = misc.imresize(misc.imread(directory + imgname[0]), img_size)

samples = x
np.random.shuffle(samples)
nb_train = 30000
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(img_size[0], img_size[1], 3))
# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(230, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


def data_generator(data, batch_size):  # 样本生成器，节省内存
    while True:
        batch = np.random.choice(data, batch_size)
        tx, ty = [], []
        for img in batch:
            # pic = Image.open(directory + img)  # .convert('L')
            # pic = misc.imresize(pic, img_size)
            # if pic.shape == (200, 200):
            #     pic = cv2.cvtColor(pic.reshape(200, 200), cv2.COLOR_GRAY2BGR)
            #     pic = misc.imresize(pic, img_size)
            pic = load_img(directory + img, target_size=(200, 200))
            pic = img_to_array(pic)
            pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
            tx.append(pic)
            ty.append(y[img])
        # for i in tx:
        #     print(i.shape)
        tyy = []
        for i in ty:
            yy = np.zeros((230,))
            yy[label_list.index(i)] = 1
            tyy.append(yy)
        ty = np.array(tyy)
        tx = preprocess_input(np.array(tx).astype(float))
        yield tx, ty


# train the model on the new data for a few epochs
model.fit_generator(data_generator(train_samples, 100), steps_per_epoch=100, epochs=1000,
                    validation_data=data_generator(test_samples, 100), validation_steps=1000)
model.save_weights('my_model_weights.h5')

# model.load_weights('my_model_weights.h5', by_name=True)
# 评价模型的全对率
from tqdm import tqdm
#
# total = 0.
# right = 0.
# step = 0
# for xp, yp in tqdm(data_generator(test_samples, 100)):
#     _ = model.predict(xp)
#     _ = np.array([i.argmax(axis=0) for i in _]).T
#     yp = np.array(yp).T
#     total += len(xp)
#     for i in range(0, len(_)):
#         if yp[_[i]][i] == 1:
#             right += 1
#     if step < 100:
#         step += 1
#     else:
#         break
#
# print('模型全对率：%s' % (right / total))


def submit_data_generator(data, path_t):
    while True:
        tx, ty = [], []
        for img in data:
            # pic = Image.open(directory + img)  # .convert('L')
            # pic = misc.imresize(pic, img_size)
            # if pic.shape == (200, 200):
            #     pic = cv2.cvtColor(pic.reshape(200, 200), cv2.COLOR_GRAY2BGR)
            #     pic = misc.imresize(pic, img_size)
            pic = load_img(path_t + img, target_size=(200, 200))
            pic = img_to_array(pic)
            pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
            tx.append(pic)
            # print(img)

        tx = preprocess_input(np.array(tx).astype(float))
        yield tx, ty


sub_x = list()
# with open('/Users/mahaoyang/Downloads/DatasetA_test_20180813/DatasetA_test/image.txt', 'r') as f:
with open('D:/lyb/DatasetA_train_20180813/DatasetA_test/image.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        sub_x.append(line)
sub_y = list()
for ix in sub_x:
    for xp, yp in tqdm(
            # submit_data_generator([ix], '/Users/mahaoyang/Downloads/DatasetA_test_20180813/DatasetA_test/test/')):
            submit_data_generator([ix], 'D:/lyb/DatasetA_train_20180813/DatasetA_test/test/')):
        ys = model.predict(xp)
        ys = np.array([i.argmax(axis=0) for i in ys]).T

        with open('submit.txt', 'a') as f:
            f.write('%s\t%s\n' % (ix, label_list[ys[0]]))
        break
