# model_left = Sequential()
# model_left.add(Dense(50, input_shape=(784,)))
# model_left.add(Activation('relu'))
#
# model_right = Sequential()
# model_right.add(Dense(50, input_shape=(784,)))
# model_rightadd(Activation('relu'))
#
# model = Sequential()
# model.add(Merge([model_left, model_right], mode='concat'))
#
# model.add(Dense(10))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit([X_train, X_train], Y_train, batch_size=64, nb_epoch=30, validation_data=([X_test, X_test], Y_test))

import numpy

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