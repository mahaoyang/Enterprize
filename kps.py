from keras.layers import Embedding

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

import os
from PIL import Image
import numpy as np

x = []
y = []
with open('train.txt', 'r') as f:
    for line in f:
        l = f.readline().strip('\n').split('\t')
        if l:
            x.append(l[0])
            y.append(l[1])
print(x, y)
directory = '/Users/mahaoyang/Downloads/DatasetA_train_20180813/train'
for imgname in os.listdir(directory):  # 参数是文件夹路径 directory

    print(imgname)

    img = Image.open(directory + imgname)
    arr = np.asarray(img, dtype=np.float32)  # 数组维度(128, 192, 3)
    print(img.size, arr.shape)
    arr = image.img_to_array(img)  # 数组维度(128, 192, 3)
    print(img.size, arr.shape)

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))
# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(x=X_train, y=Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val), callbacks=[es])
