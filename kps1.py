from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.applications import VGG16, VGG19
import keras
from keras.preprocessing.image import load_img, img_to_array

# mnist attention
import numpy as np
import pickle

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
tx, ty = [], []

with open('D:/lyb/DatasetA_train_20180813/attributes_per_class.txt', 'r') as f:
    class_atri = dict()
    for i in f.readlines():
        ii = i.strip('\n').split('\t')
        class_atri[ii[0]] = ii[1:]

with open('D:/lyb/DatasetA_train_20180813/attribute_list.txt', 'r') as f:
    atri_s = dict()
    for i in f.readlines():
        ii = i.strip('\n').split('\t')
        atri_s[int(ii[0])] = ii[1:]

wtx = []
for i in x:
    wtx.append(class_atri[y[i]])
wtx = np.array(wtx)

# for img in x:
#     # pic = Image.open(directory + img)  # .convert('L')
#     # pic = misc.imresize(pic, img_size)
#     # if pic.shape == (200, 200):
#     #     pic = cv2.cvtColor(pic.reshape(200, 200), cv2.COLOR_GRAY2BGR)
#     #     pic = misc.imresize(pic, img_size)
#     pic = load_img('D:/lyb/DatasetA_train_20180813/train/' + img, target_size=(64, 64))
#     pic = img_to_array(pic)
#     pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
#     tx.append(pic)

# with open('imd.pickle', 'wb') as f:
#     pickle.dump(tx, f)
with open('imd.pickle', 'rb') as f:
    tx = pickle.load(f)

np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_DIM = 28
lstm_units = 64


# data pre-processing
# (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
# X_train = X_train.reshape(-1, 28, 28) / 255.
# X_test = X_test.reshape(-1, 28, 28) / 255.
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)


# first way attention
def attention_3d_block(inputs):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# inputs = Input(shape=(TIME_STEPS, INPUT_DIM), name='word_input')
# drop1 = Dropout(0.3)(inputs)
# lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
# attention_mul = attention_3d_block(lstm_out)
# attention_flatten = Flatten()(attention_mul)
# drop2 = Dropout(0.3)(attention_flatten)
# output = Dense(10, activation='sigmoid')(drop2)
# model = Model(inputs=inputs, outputs=output)

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = VGG19(include_top=False, weights=None, input_shape=(64, 64, 3))

# # Now let's get a tensor with the output of our vision model:
# image_input = Input(shape=(64, 64, 3), name='img_input')
# encoded_image = vision_model(image_input)

# # Next, let's define a language model to encode the question into a vector.
# # Each question will be at most 100 word long,
# # and we will index words as integers from 1 to 9999.
# question_input = Input(shape=(30,30,), dtype='float')
# # embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=30)(question_input)
# drop1 = Dropout(0.3)(question_input)
# lstm_out = Bidirectional(LSTM(230, input_shape=wtx.shape, return_sequences=True), name='bilstm')(drop1)
# attention_mul = attention_3d_block(lstm_out)
# attention_flatten = Flatten()(attention_mul)
# drop2 = Dropout(0.3)(attention_flatten)
# encoded_question = Dense(256, activation='sigmoid')(drop2)

# # Let's concatenate the question vector and the image vector:
# merged = keras.layers.concatenate([encoded_question, encoded_image])

# # And let's train a logistic regression over 1000 words on top:
# output = Dense(230, activation='softmax')(merged)
#
# # This is our final model:
# vqa_model = Model(inputs=[image_input, question_input], outputs=output)

xm = vision_model.output
xm = Flatten()(xm)
xm = Dense(30, activation='relu')(xm)
out = Dense()

model = Model(inputs=vision_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

# print(img)
# The next stage would be training this model on actual data.
y = np.array(y)
model.fit(x=[tx, wtx], y=y, validation_split=0.2, epochs=20, batch_size=200, verbose=2)
model.save_weights('lstm.h5')
