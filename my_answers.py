import string

import numpy as np

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
import keras


def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    i = 0
    while i < len(series) - window_size:
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
        i += 1

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alphabet = list(string.ascii_lowercase)
    allowed_chars = set(punctuation + alphabet + [' '])

    all_chars = set(text)
    chars_to_remove = all_chars - allowed_chars

    for char_to_remove in chars_to_remove:
        text = text.replace(char_to_remove,' ')

    return text


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = 0
    while i < len(text) - window_size:
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size

    return inputs, outputs


def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model