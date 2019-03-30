# import numpy as np # linear algebra
# from numpy import newaxis
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM, GRU
# from keras.models import Sequential
# from keras import optimizers
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# from HelpingFunctions import *
#
# plt.style.use('fivethirtyeight')
# print('import completed')
# Enrol_window = 100
# print('enrol window set to',Enrol_window )
#
# print('Support functions defined')
#
# dataset = pd.read_csv('./PMChineFiveCitie/ShenyangECXELPM20100101_20151231---V2).csv',sep=";")
# # print('corrrr:',dataset.corr(method ='pearson'))
#
# print(dataset.head())
#
# dat=pd.to_datetime(dataset["date"])
# date =np.array(dat)
#
# one=np.array(dataset["PM_Taiyuanjie"])
# two= np.array(dataset["PM_US Post"])
# three = np.array(dataset["PM_Xiaoheyan"])
# # PM_Shenyang = np.column_stack((date,(one+two+three)/3))
# PM_Shenyang = np.array((one+two+three)/3)
#
#
# newPMdf = pd.DataFrame(data = {'PM_Shenyang':PM_Shenyang},index = date)
# print(newPMdf)
#
# # feature_train, label_train, feature_test, label_test = load_data(dataset, 'PM_Taiyuanjie', Enrol_window, True)
# feature_train, label_train, feature_test, label_test = load_data(newPMdf, 'PM_Shenyang', Enrol_window, True)
#
# newPMdf["PM_Shenyang"][:'2015'].plot(figsize=(16,4),legend=True)
# # newPMdf["PM_Shenyang"]['2015':].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data
# plt.legend(['Training set','Test set'])
# plt.title('True data')
# plt.show()
#
# model = Sequential() #https://keras.io/layers/recurrent/#lstm0000000000000000
# model.add(LSTM(50, return_sequences=True, input_shape=(feature_train.shape[1],1)))  #https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
#
# model.add(Dropout(0.2))
# model.add(LSTM(100, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation = "linear"))
#
# opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print ('model compiled')
#
# model.fit(feature_train, label_train, batch_size=512, epochs=5, validation_data = (feature_test, label_test))
# # model.fit(feature_train, label_train, batch_size=512, epochs=5, validation_split = 0.1)
# predicted = model.predict(feature_test)
# # loss, accuracy = model.evaluate(feature_test, label_test)
# # print("loss",loss, "acc",accuracy)
# plot_results(predicted,label_test)
# # l=[]
# # a=[]
# # loss =np.array(l,a)
# # accuracy=np.array(a)
# loss_acc = model.evaluate(feature_test, predicted)
# print("loss accuracy",loss_acc)
# # arr = []
# # scores = np.array(arr)
#
# # scores = model.evaluate(feature_test,label_test)
# # print("%s: %.2f%%" % (model.metrics_names, scores[1]*100))
# # print("%s: %.2f%%" % (model.metrics_names, scores[1]*100))
# # cvscores = []
# # cvscores.append(scores[1] * 100)
# # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# # plot_results_multiple(predicted,label_test)
import sys

import numpy as np # linear algebra
from keras.layers import BatchNormalization
from numpy import newaxis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from HelpingFunctions import *
plt.style.use('fivethirtyeight')

print ('import completed')

Enrol_window = 100

print ('enrol window set to',Enrol_window )

print('Support functions defined')

dataset = pd.read_csv('./PMChineFiveCitie/ShenyangECXELPM20100101_20151231---V2).csv',sep=";")
# dataset = pd.read_csv('./PMChineFiveCitie/ShenyangECXELPM20100101_20151231---V2).csv',sep=";", index_col='date', parse_dates=['date'])
# sep=";",
# print('corrrr:',dataset.corr(method ='pearson'))


print(dataset.head())



dat=pd.to_datetime(dataset["date"])
date =np.array(dat)

one=np.array(dataset["PM_Taiyuanjie"])
one1 = one.tolist()
two= np.array(dataset["PM_US Post"])
two1 = two.tolist()
three = np.array(dataset["PM_Xiaoheyan"])
three1 = three.tolist()
# one[(one == 0)&(two != 0) & (three != 0)] = (two + three)/2
# PM_Shenyang = np.array((one+two+three)/3)
PM_Shenyang = []
for i in range(len(one1)):
    if (one1[i] == 0 and two1[i] != 0 and three1[i] != 0) or (one1[i] != 0 and two1[i] == 0 and three1[i] != 0) or (
            one1[i] != 0 and two1[i] != 0 and three1[i] == 0):

        PM_Shenyang.append((one1[i] + two1[i] + three1[i])/2)

    elif (one1[i] != 0 and two1[i] != 0 and three1[i] != 0):
        PM_Shenyang.append((one1[i] + two1[i] + three1[i]) / 3)

    elif (one1[i] == 0 and two1[i] == 0 and three1[i] != 0) or (one1[i] == 0 and two1[i] != 0 and three1[i] == 0) or (
            one1[i] != 0 and two1[i] == 0 and three1[i] == 0):
        PM_Shenyang.append(one1[i] + two1[i] + three1[i])

    elif (one1[i] == 0 and two1[i] == 0 and three1[i] == 0) or (one1[i] == 0 and two1[i] == 0 and three1[i] == 0) or (
            one1[i] == 0 and two1[i] == 0 and three1[i] == 0):
        PM_Shenyang.append(0)

PM_Shenyang1 = np.array(PM_Shenyang)
PM_Shenyang1[PM_Shenyang1 > 750] = 400
print("PMnumbers", PM_Shenyang)
# PM_Shenyang = np.column_stack((date,(one+two+three)/3))
# PM_Shenyang = np.array((one+two+three)/3)
newPMdf = pd.DataFrame(data = {'PM_Shenyang':PM_Shenyang1},index = date)
print(newPMdf)

# feature_train, label_train, feature_test, label_test = load_data(dataset, 'PM_Taiyuanjie', Enrol_window, True)
feature_train, label_train, feature_test, label_test = load_data(newPMdf, 'PM_Shenyang', Enrol_window, True)
newPMdf["PM_Shenyang"][:'2015'].plot(figsize=(16,4),legend=True)
newPMdf["PM_Shenyang"]['2015':].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data
plt.legend(['Training set','Test set'])
plt.title('True data')
plt.show()


model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1, feature_train.shape[0],1)))
# model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(Dense(10,kernel_initializer='normal', activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

print('model compiled')

model.fit(feature_train, label_train, batch_size=512, epochs=5, validation_data = (feature_test, label_test))



predicted = model.predict(feature_test)
plot_results(predicted,label_test)