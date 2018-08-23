import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
from process_data import *

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=2048, input_dim=4096, kernel_initializer="normal", activation="relu"))

model.add(Dropout(0.2))
model.add(Dense(units=1024, kernel_initializer="normal", activation="relu"))

model.add(Dropout(0.3))
model.add(Dense(units=512, kernel_initializer="normal", activation="relu"))

model.add(Dense(units=149, kernel_initializer="normal", activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(shuffle=True)

train_history = model.fit(x=X_train_data, y=y_train_data, validation_split=0.2, epochs=20, batch_size=200, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train history")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['train', 'validation'], loc="upper left")
    plt.show()


show_train_history(train_history, "acc", "val_acc")


def load_one_hot_to_label(encoder, data):
    num = None
    for i in range(len(data)):
        if data[i] == 1:
            num = i
    label = encoder.inverse_transform(num)
    print("label:" + label)
    return label


model.predict_classes(X_train_data)

test_features = load(open(path + "tianchi_test_features.pkl", 'rb'))
test_data = []
keys = []
for key, label in test_features.items():
    test_data.append(test_features[key][0])
    keys.append(key)
teat_predict_data = np.array(test_data)

predict_data = model.predict_classes(teat_predict_data)

predict_data_list = list(predict_data)

labels_list = []
for i in range(len(predict_data_list)):
    label = encoder.inverse_transform(predict_data_list[i])
    labels_list.append(label)

with open("submit.txt", "w") as f:
    for i in range(len(keys)):
        f.write(keys[i] + ".jpeg" + "\t" + labels_list[i] + "\n")
