from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.applications import VGG16, VGG19
import keras

# mnist attention
import numpy as np

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
(X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
X_train = X_train.reshape(-1, 28, 28) / 255.
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# first way attention
def attention_3d_block(inputs):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
drop1 = Dropout(0.3)(inputs)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(10, activation='sigmoid')(drop2)
model = Model(inputs=inputs, outputs=output)

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = VGG19(include_top=False, weights=None, input_shape=(64, 64, 3))

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(64, 64, 3))
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
drop1 = Dropout(0.3)(embedded_question)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)
encoded_question = Dense(256, activation='sigmoid')(drop2)


# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# And let's train a logistic regression over 1000 words on top:
output = Dense(230, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# The next stage would be training this model on actual data.
vqa_model.fit(data_generator(train_samples, 100), steps_per_epoch=1000, epochs=10,
                    validation_data=data_generator(test_samples, 100), validation_steps=100)