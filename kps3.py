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
weights = 'vgg19.h5'

# path = 'D:/lyb/'
path = '/Users/mahaoyang/Downloads/'


class NN(object):
    def __init__(self, base_path, model_weights):
        self.base_path = base_path
        self.model_weights = model_weights

    @staticmethod
    def model():
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
            test_list[i]['label_array'] = model.predict(test_list[i]['img_array'])
            test_list[i]['label'] = data['label_map'][np.max(test_list[i]['label_array'])]
            submit_lines.append([i, test_list[i]['label']])

        for i in submit_lines:
            with open('submit.txt', 'a') as f:
                f.write('%s\t%s\n' % (i[0], i[1]))


if __name__ == '__main__':
    nn = NN(base_path=path, model_weights=weights)
    nn.train()
    nn.submit()
