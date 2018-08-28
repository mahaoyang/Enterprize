from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.layers import Input
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Activation, Dropout, Embedding
import numpy as np
import pickle
from data2array import data2array

img_size = (64, 64, 3)

inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
base_model = VGG19(input_tensor=inputs, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(230, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# base_path = 'D:/lyb/'
base_path = '/Users/mahaoyang/Downloads/'
data = data2array(base_path)
train_list = data['train_list']
x = []
y = []
for i in train_list:
    x.append(train_list[i]['img_array'])
    y.append(train_list[i]['label_array'])

model.fit(x=x, y=y, validation_split=0.2, epochs=20, batch_size=200, verbose=2)
model.save('vgg19.h5')
