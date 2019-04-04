import numpy as np  # linear algebra
from numpy import newaxis
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

sc = MinMaxScaler(feature_range=(0, 1))


# sc = MinMaxScaler()

def load_data(datasetname, column, seq_len,
              normalise_window):  # it BRINGS: 1: The data, 2: Column Name to work with, 3: Just a number of Window (100), 4:Boolean value
    # A support function to help prepare datasets for an RNN/LSTM/GRU
    data = datasetname.loc[:, column].replace(0,
                                              1)  # Take the dataframe and get all the data from the column we want to work with and make zeros ones
    sequence_length = seq_len + 1  # make  length value greater than 1

    result = []  # https://machinelearningmastery.com/sequence-prediction/
    for index in range(len(data) - sequence_length):  # Pare thn  krata ta data tou column xoris to window
        result.append(data[
                      index: index + sequence_length])  # https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
    print('Values', 0.7 * np.array(result))
    if normalise_window:
        result = normalise_windows(result)
        result = sc.fit_transform(result)

    result = np.array(result)
    train_size = int(len(result) * 0.66)
    train, test = result[0:train_size], result[train_size:len(result)]

    # testsplit = int(len(test)* 0.55)
    # tests, val = test[0:testsplit] , test[testsplit:len(test)]

    print('Data: %d' % (len(result)))
    print('Training Dataset: %d' % (len(train)))
    print('Testing Dataset: %d' % (len(test)))
    # print('Test Validation: %d' % (len(val)))

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]

    # Last 10% is used for validation test, first 90% for training

    x_train = np.reshape(x_train, (
    x_train.shape[0], x_train.shape[1], 1))  # https://www.hackerrank.com/challenges/np-shape-reshape/problem
    print(x_train.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))  # https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

    return [x_train, y_train, x_test, y_test, train_size, result]


def normalise_windows(window_data):
    # A support function to normalize a dataset
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def plot_results(predicted_data, true_data):
    # Standard plot /////////////////////////////////
    fig = plt.figure(facecolor='white', figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend(['True Data', 'Prediction'])
    plt.show()


print('Support functions defined')