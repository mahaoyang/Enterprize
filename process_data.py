from pickle import load
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from random import sample


def load_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_label(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_label = tokens[0], tokens[1]
        image_id = image_id.split('.')[0]
        if image_id not in mapping:
            mapping[image_id] = ""
        mapping[image_id] = image_label
    return mapping


def load_labels_features(pic_features, dataset):
    features = {k: pic_features[k] for k in dataset}
    return features


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(sample(dataset, len(dataset)))


def create_sequences(labels, photos):
    X1, y = list(), list()
    for key, label in labels.items():
        X1.append(photos[key][0])
        y.append(label)
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y)
    y_data = to_categorical(encoded_Y)
    labels = y

    return np.array(X1), y_data, labels, encoder, encoded_Y


path = "../../data/"
dataset = load_set(path + "train.txt")
doc = load_doc(path + "train.txt")
lable_data = load_label(doc)
pic_features = load_features(path + "tianchi_features.pkl", dataset)
label_features = load_labels_features(lable_data, dataset)
X_train_data, y_train_data, labels, encoder, encoded_Y = create_sequences(label_features, pic_features)
