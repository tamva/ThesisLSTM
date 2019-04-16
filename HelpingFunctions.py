import numpy as np # linear algebra
from numpy import newaxis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

sc = MinMaxScaler(feature_range=(0, 1))



#it BRINGS: 1: The data, 2: Column Name to work with, 3: Just a number of Window (100), 4:Boolean value
def data_loader(datasetname, column, seq_len, normalise_window): 
    #A support function for preparing data sets for an LSTM network
    #Take the dataframe and get all the data from the column we want to work with and replace them from 0 to 1 in order to make the division 
    data = datasetname.loc[:, column].replace(0, 1)
    sequence_length = seq_len + 1 # make  length value greater than 1

    result = []                               
    for index in range(len(data) - sequence_length): #Pare thn  krata ta data tou column xoris to window
        result.append(data[index: index + sequence_length])#https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
    
    if normalise_window:
        result = normalise_windows(result)
        result = sc.fit_transform(result)

    result = np.array(result)
    X = result
    train_size = int(len(X) * 0.66)
    train, test= X[0:train_size], X[train_size:len(X)]

    #testsplit = int(len(test)* 0.55)
    #tests, val = test[0:testsplit] , test[testsplit:len(test)]
     
    print('Data: %d' % (len(X)))
    print('Training Dataset: %d' % (len(train)))
    print('Testing Dataset: %d' % (len(test)))
    #print('Test Validation: %d' % (len(val)))  
    

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
     

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #https://www.hackerrank.com/challenges/np-shape-reshape/problem
    print(x_train.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
    
    return [x_train, y_train, x_test, y_test, train_size, X]


def normalise_windows(window_data):
    # Prepare the dataset, note that the PM  price data will be normalized between 0 and 1
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def plot_results(predicted_data, true_data):
    # Standard plot
    fig = plt.figure(facecolor='white',figsize=(16,4))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data,label='Prediction')
    plt.legend(['True Data', 'Prediction'])
    plt.show()

print('Support functions defined')