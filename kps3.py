from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, concatenate
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.applications import VGG16, VGG19
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras import metrics
import numpy as np
import pickle
from data2array import data2array

img_size = (64, 64, 3)
weights = 'vgg19_pw.h5'

# path = 'D:/lyb/'


path = '/Users/mahaoyang/Downloads/'


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
            max_index = int(np.where(test_list[i]['label_array'] == np.max(test_list[i]['label_array']))[1][0])
            test_list[i]['label'] = data['label_map'][max_index]
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
    # word_input = Embedding(input_dim=300, output_dim=230, input_length=1)(word_input)
    # word_input = Flatten()(word_input)
    merged = concatenate([word_input, img_features])

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


def model_pw():
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
    base_model = VGG19(input_tensor=inputs, weights='imagenet', include_top=False)
    x = base_model.output
    x = BatchNormalization(epsilon=1e-6, weights=None)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(300)(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    # opti = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opti = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=opti, loss=losses.mean_squared_error,
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    model.summary()
    return model


def distance(vec1, vec2):
    sub = np.square(np.array(vec1) - np.array(vec2).astype('float32'))
    # print(vec1)
    return np.sqrt(np.sum(sub))


def euclidean_distances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ed = np.sqrt(SqED)
    return ed


class PWNN(SimpleNN):
    @staticmethod
    def model():
        return model_pw()

    def train(self):
        model = self.model()
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        x = []
        y = []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            y.append(train_list[i]['label_real_name_class_wordembeddings'])
        x = np.array(x)
        y = np.array(y)

        model.fit(x=x, y=y, validation_split=0.2, epochs=200, batch_size=200)
        model.save(self.model_weights)

        model.evaluate(x=x[train_num:], y=y[train_num:], batch_size=200)
        return model

    def submit(self):
        model = self.model()
        data = data2array(self.base_path)
        reverse_label_list = {v: k for k, v in data['label_list'].items()}
        test_list = data['test_list']
        test_list_array = []
        model.load_weights(self.model_weights)
        submit_lines = []
        for i in test_list:
            test_list_array.append(test_list[i]['img_array'])
        test_list_label_array = model.predict(np.array(test_list_array))
        class_wordembed_keys = list(data['class_wordembeddings'].keys())
        class_wordembed_array = np.array(list(data['class_wordembeddings'].values())).astype('float32')
        dist = euclidean_distances(test_list_label_array, class_wordembed_array)
        distance_all = []
        for i in dist:
            i =list(i)[0]
            ii = i.index(min(i))
        for key in class_wordembed:
            distance_all.append(distance(test_list[i]['label_array'], np.array(data['class_wordembeddings'][key])))
        most_like = class_wordembed[distance_all.index(min(distance_all))]
        test_list[i]['label'] = reverse_label_list[most_like]
        submit_lines.append([i, test_list[i]['label']])

        for i in submit_lines:
            with open('submit.txt', 'a') as f:
                f.write('%s\t%s\n' % (i[0], i[1]))


if __name__ == '__main__':
    # nn = SimpleNN(base_path=path, model_weights=weights)
    # nn.train()
    # nn.submit()
    # nn = MixNN(base_path=path, model_weights=weights)
    # nn.train()
    nn = PWNN(base_path=path, model_weights=weights)
    # nn.train()
    nn.submit()
