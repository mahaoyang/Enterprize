from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, concatenate
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
weights = 'vgg19.h5'

path = 'D:/lyb/'
# path = '/Users/mahaoyang/Downloads/'


def model_cnn():
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
    base_model = VGG19(input_tensor=inputs, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(230, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()
    return model


class SimpleNN(object):
    def __init__(self, base_path, model_weights):
        self.base_path = base_path
        self.model_weights = model_weights

    @staticmethod
    def model():
        return model_cnn()

    def train(self):
        model = self.model()
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        x = []
        y = []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            temp = np.zeros((230,))
            temp[train_list[i]['label_array']] = 1
            y.append(temp)
        x = np.array(x)
        y = np.array(y)

        model.fit(x=x[:train_num], y=y[:train_num], validation_split=0.2, epochs=20, batch_size=200)
        model.save(self.model_weights)

        model.evaluate(x=x[train_num:], y=y[train_num:])
        return model

    def submit(self):
        model = self.model()
        data = data2array(self.base_path)
        test_list = data['test_list']
        model.load_weights(self.model_weights)
        submit_lines = []
        for i in test_list:
            test_list[i]['label_array'] = model.predict(np.array([test_list[i]['img_array']]))
            max_index = np.where(test_list[i]['label_array'] == np.max(test_list[i]['label_array']))
            test_list[i]['label'] = data['label_map'][max_index[0]]
            submit_lines.append([i, test_list[i]['label']])

        for i in submit_lines:
            with open('submit.txt', 'a') as f:
                f.write('%s\t%s\n' % (i[0], i[1]))


def model_mix():
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
    base_model = VGG19(input_tensor=inputs, weights='imagenet', include_top=False)
    x = base_model.output
    img_features = Flatten()(x)

    word_input = Input(shape=(300,), dtype='float32')
    embeded = Embedding(input_dim=300, output_dim=230, input_length=1)(word_input)
    embeded = Flatten()(embeded)
    merged = concatenate([embeded, img_features])

    predictions = Dense(230, activation='softmax')(merged)

    model = Model(inputs=[base_model.input, word_input], outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()
    return model


class MixNN(SimpleNN):
    @staticmethod
    def model():
        return model_mix()

    def train(self):
        model = self.model()
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        x = []
        wx = []
        y = []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            wx.append(train_list[i]['label_real_name_class_wordembeddings'])
            temp = np.zeros((230,))
            temp[train_list[i]['label_array']] = 1
            y.append(temp)
        x = np.array(x)
        wx = np.array(wx)
        y = np.array(y)

        model.fit(x=[x[:train_num], wx[:train_num]], y=y[:train_num], validation_split=0.2, epochs=20, batch_size=200)
        model.save(self.model_weights)

        model.evaluate(x=[x[train_num:], wx[train_num:]], y=y[train_num:])
        return model


if __name__ == '__main__':
    nn = SimpleNN(base_path=path, model_weights=weights)
    nn.train()
    nn.submit()
    # nn = MixNN(base_path=path, model_weights=weights)
    # nn.train()
