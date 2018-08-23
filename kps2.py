from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.layers import Input
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
print(wtx.shape)

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

input1 = Input((64, 64, 3), name='i1')
vision_model = VGG19(include_top=False, weights=None, input_shape=(64, 64, 3))
xm = vision_model.output
xm = Flatten()(xm)
xm = Dense(256, activation='relu')(xm)
input2 = Input(shape=wtx.shape, name='i2')
embedded_question = Embedding(input_dim=wtx.shape[0], output_dim=256, input_length=1)(input2)
wtxl = LSTM(256)(embedded_question)
merged = keras.layers.concatenate([wtxl, xm])
out = Dense(230, activation='softmax')(merged)

model = Model(inputs=[vision_model.input, input2], outputs=out)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
y = np.array(y)
model.fit(x=[tx, wtx], y=y, validation_split=0.2, epochs=20, batch_size=200, verbose=2)
model.save_weights('lstm.h5')
