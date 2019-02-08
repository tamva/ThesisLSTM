import numpy as np # linear algebra
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
two= np.array(dataset["PM_US Post"])
three = np.array(dataset["PM_Xiaoheyan"])
# PM_Shenyang = np.column_stack((date,(one+two+three)/3))
PM_Shenyang = np.array((one+two+three)/3)
newPMdf = pd.DataFrame(data = {'PM_Shenyang':PM_Shenyang},index = date)
print(newPMdf)

# feature_train, label_train, feature_test, label_test = load_data(dataset, 'PM_Taiyuanjie', Enrol_window, True)
feature_train, label_train, feature_test, label_test = load_data(newPMdf, 'PM_Shenyang', Enrol_window, True)

newPMdf["PM_Shenyang"][:'2015'].plot(figsize=(16,4),legend=True)
newPMdf["PM_Shenyang"]['2015':].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data
plt.legend(['Training set','Test set'])
plt.title('True data')
plt.show()


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(feature_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "linear"))

model.compile(loss='mse', optimizer='adam')

print ('model compiled')

model.fit(feature_train, label_train, batch_size=512, epochs=5, validation_data = (feature_test, label_test))



predicted = model.predict(feature_test)
plot_results(predicted,label_test,date)