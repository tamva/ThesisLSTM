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
from HelpingFunctions import * #we import our Helping Function file that contains all the functions
from sklearn.metrics import r2_score

#Here is the style that we use in our plots
plt.style.use('fivethirtyeight')
print ("Import completed")

# Enter the steps to register on the network.
#RNN / LSTM / GRU can be taught at times times as large as the number of times you register them, #and no longer (a fundamental limitation).
# So by design these networks are deep / long to catch repeating motifs.
Enrol_window = 100

print ("Enrol window set to",Enrol_window )
print("Support functions defined")

#Load the data
dataset = pd.read_csv('./Data/ShenyangECXELPM20100101_20151231---V2).csv',sep=";")

print(dataset.head())

#Create the list which contains the date data
dat=pd.to_datetime(dataset["date"])
date =np.array(dat)

#Here we create three arrays with all PM station measurement of Shenyang and covert them to lists #so then we can edit them one by one
one=np.array(dataset["PM_Taiyuanjie"])
one1 = one.tolist()
two= np.array(dataset["PM_US Post"])
two1 = two.tolist()
three = np.array(dataset["PM_Xiaoheyan"])
three1 = three.tolist()


#In order to calculate the average value of Shenyang we take the inputs of each station that have non #zero values 
#and divide them with the sum of the station that has non zero values 
PM_Shenyang = []
for i in range(len(one1)):
    #Here we calculate the average if one station has 0 measurment 
    if (one1[i] == 0 and two1[i] != 0 and three1[i] != 0) or (one1[i] != 0 and two1[i] == 0 and three1[i] != 0) or ( one1[i] != 0 and two1[i] != 0 and three1[i] == 0):
        PM_Shenyang.append((one1[i] + two1[i] + three1[i])/2)

    #Here we calculate the average if there is no zero values in station
    elif (one1[i] != 0 and two1[i] != 0 and three1[i] != 0):
        PM_Shenyang.append((one1[i] + two1[i] + three1[i]) / 3)

    #Here we calculate the average if 2 of the station has zero values
    elif (one1[i] == 0 and two1[i] == 0 and three1[i] != 0) or (one1[i] == 0 and two1[i] != 0 and
               three1[i]== 0) or (one1[i] != 0 and two1[i] == 0 and three1[i] == 0):
         PM_Shenyang.append(one1[i] + two1[i] + three1[i])

    #Here we calulate the average if 3 of the station has zero values
    elif (one1[i] == 0 and two1[i] == 0 and three1[i] == 0) or (one1[i] == 0 and two1[i] == 0 and three1[i]   
            == 0) or (one1[i] == 0 and two1[i] == 0 and three1[i] == 0):
        PM_Shenyang.append(0)

PM_Shenyang1 = np.array(PM_Shenyang)

#In order to overtake the error values we replace all the values over 750 with the number 400
PM_Shenyang1[PM_Shenyang1 > 750] = 400
print("Average PM Values Completed")

#This is the finale dataframe that contains all the values for the prediction 
newPMdf = pd.DataFrame(data = {'PM_Shenyang':PM_Shenyang1},index = date)
print("Final Data Frame: ", newPMdf)
print("Start data preparation for training....")

# Prepare the dataset, note that the PM 2.5 values for PM_Shenyang data will be normalized between 0 and 1
# A feature is an input variable, in this case a PM 2.5 Value
# Selected 'PM_Shenyang' PM 2.5 Values
feature_train, label_train, feature_test, label_test, train_size, X = data_loader(newPMdf, 'PM_Shenyang', Enrol_window, True)
newPMdf["PM_Shenyang"][0:train_size].plot(figsize=(16,4),legend=True)
newPMdf["PM_Shenyang"][train_size:len(X)].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data
plt.legend(['Training dataset','Testing dataset'])
plt.title("True data")
plt.show()

#The LSTM model we will test, let’s see if the sinus prediction results can be matched
#Here we start the model building structure 
model = Sequential()
model.add(LSTM(50, kernel_initializer='normal', return_sequences=True, input_shape=(feature_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(100, kernel_initializer='normal', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, kernel_initializer='normal', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,kernel_initializer='normal', activation='linear'))

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

print("Model compiled")

#Train the model with Keras fit function
model.fit(feature_train, label_train, batch_size=300, epochs=5, validation_split = 0.2)

#Here we use the model and predict the PM2.5 of Shenyang
predicted = model.predict(feature_test)
plot_results(predicted,label_test)

#Here we do the evaluation test
kappa = model.evaluate(feature_test,label_test)

print("Evaluation Test: ", kappa)

#Here we calculate the R^2 score of training kai testing set on our model
print("Start R2 Score for taining and testing set...")
feature_train_pred = model.predict(feature_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(label_train, feature_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(label_test, predicted)))
