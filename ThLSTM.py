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

dataset = pd.read_csv('./ShenyangECXELPM20100101_20151231---V2).csv',sep=";")

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

#
PM_Shenyang1[PM_Shenyang1 > 750] = 400
print("PMnumbers", PM_Shenyang)
newPMdf = pd.DataFrame(data = {'PM_Shenyang':PM_Shenyang1},index = date)
print(newPMdf)

# feature_train, label_train, feature_test, label_test = load_data(dataset, 'PM_Taiyuanjie', Enrol_window, True)
feature_train, label_train, feature_test, label_test, train_size, X = load_data(newPMdf, 'PM_Shenyang', Enrol_window, True)
newPMdf["PM_Shenyang"][0:train_size].plot(figsize=(16,4),legend=True)
newPMdf["PM_Shenyang"][train_size:len(X)].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data
plt.legend(['Training set','Test set'])
plt.title('True data')
plt.show()

model = Sequential()
model.add(LSTM(50, kernel_initializer='normal', return_sequences=True, input_shape=(feature_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(100, kernel_initializer='normal', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, kernel_initializer='normal', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,kernel_initializer='normal', activation='linear'))

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

print('model compiled')

model.fit(feature_train, label_train, batch_size=300, epochs=5, validation_split = 0.2)


#Edw kanoume to prediction
predicted = model.predict(feature_test)
plot_results(predicted,label_test)

#Edw kanoume to evaluation test
kappa = model.evaluate(feature_test,label_test)
print('Evaluation Test: ', kappa)
