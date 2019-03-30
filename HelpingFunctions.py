import numpy as np # linear algebra
from numpy import newaxis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU

from keras.models import Sequential

from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

sc = MinMaxScaler(feature_range=(0, 1))
# sc = MinMaxScaler()

def load_data(datasetname, column, seq_len, normalise_window): #it BRINGS: 1: The data, 2: Column Name to work with, 3: Just a number of Window (100), 4:Boolean value
    # A support function to help prepare datasets for an RNN/LSTM/GRU
    data = datasetname.loc[:, column].replace(0, 1)#Take the dataframe and get all the data from the column we want to work with and make zeros ones
    sequence_length = seq_len + 1 # make  length value greater than 1

    result = []                                 #https://machinelearningmastery.com/sequence-prediction/
    for index in range(len(data) - sequence_length): #Pare thn  krata ta data tou column xoris to window
        result.append(data[index: index + sequence_length])#https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
    print('Values', 0.9 * np.array(result))
    if normalise_window:
        result = normalise_windows(result)
        result = sc.fit_transform(result)

    result = np.array(result)

    # Last 10% is used for validation test, first 90% for training
    row = round(0.9 * result.shape[0])

    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    print(x_train,x_test,y_train,y_test,)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #https://www.hackerrank.com/challenges/np-shape-reshape/problem
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    # A support function to normalize a dataset
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def plot_results(predicted_data, true_data):
    # Standard plot /////////////////////////////////
    fig = plt.figure(facecolor='white',figsize=(16,4))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data,label='Prediction')
    plt.legend(['True Data', 'Prediction'])
    plt.show()

    # #SECONDARY PLOT /////////////////////////
    # fig = plt.figure(facecolor='white',figsize=(16,4))
    # ax = fig.add_subplot(111)
    # ax.plot(true_data, label='True Data')
    # plt.plot(predicted_data, label='Prediction')
    # locs,labels = plt.xticks()
    # labels = ['2012-06','2013-01','2013-06','2014-01','2014-06','2015-01','2015-12']
    # plt.xticks(locs, labels)
    # plt.show()


# def plot_results_multiple(predicted_data, true_data):
#     fig = plt.figure(facecolor='white')
#     ax = fig.add_subplot(111)
#     ax.plot(true_data, label='True Data')
#     #Pad the list of predictions to shift it in the graph to it's correct start
#     # for i, data in enumerate(predicted_data):
#     data=10
#     padding = predicted_data[-1]
#     plt.plot(padding + data, label='Prediction')
#     plt.legend()
#     plt.show()
print('Support functions defined')