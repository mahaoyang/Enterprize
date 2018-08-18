import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

[x_train, y_train], (x_test, y_test) = mnist.load_data()
print(x_train.shape)
plt.imshow(x_train[0])

