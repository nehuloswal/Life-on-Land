#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""Team Members: Nehul Oswal, Fedrik Durao, Anirban Bhattacharya
This code is used to make a prediction model using LSTMs, that can predict future AQI
values of a state, based on its history of AQI values and weather conditions. Such weather parameters
are wind speed, percent humidity, temperature, and pressure. The same model is trained and tested for all the unique
States (by State Code). The data for each state will have 5 columns: Date, AQI, Wind speed, Temp, Humidity,
and Pressure. Pandas is used to concatenate all the data by year for each parameter, then concatenate the
resulting parameter files. And then split the files by state, each one being ran on the LSTM model in a loop.
Data was obtained @ 'https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw.....check'
Datapipeline: keras and pandas
Concept: Using an LSTM time series to make predictions
Machine: Google Cloud Dataproc running Ubuntu"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.optimizers import Adam
import pandas as pd
import glob
import numpy as np

# convert series to supervised learning
def series_to_supervised(data):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(1, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg


subpath = r'dataForLSTMmodel'#the path in which most files are stored
folders=['\\aqi', '\\wind', '\\temp', '\\press', '\\rh']#seperate folder directories of seperated data

ext=''#extension used later while saving files
for folder in folders:#goes through folders one at a time
    path=subpath+folder#path of folder
    print(path)
    all_files = glob.glob(path + "/*.zip")#gets all zip files at the directory
    li = []
    keepcols=['State Code', 'County Code', 'Date', 'AQI', 'Arithmetic Mean', 'Date Local']#columns we want to keep

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)#loops through all zip files
        #the extension of the upcoming saved file depends on which data is being read
        #and depending on the data, some rows we will ignore
        if('WIND' in filename):
            ext='WIND'
            df = df.loc[(df['Parameter Code'] == 61103)]#keep wind speed rows
        elif('TEMP' in filename):
            ext='TEMP'
        elif('PRESS' in filename):
            ext='PRESS'
        elif('RH' in filename):
            ext='RH'
            df = df.loc[(df['Parameter Code'] == 62201)]#keep relative humidity rows
        else:
            ext='AQI'
        for col in df.columns:
            if(col not in keepcols):#remove all unwanted columns
                df.drop(col, axis=1, inplace=True)
        df.rename(columns={'Date Local':'Date'}, inplace=True)#standardize the naming convention of 'dates' column
        if('WIND' in filename or 'TEMP' in filename or 'PRESS' in filename or 'RH' in filename):
            df.drop('AQI', axis=1, inplace=True)#the weather parameter files have non helpful AQI columns. Now removed
        li.append(df)#store data frame for later

    frame = pd.concat(li, axis=0, ignore_index=True)#concatenate all dataframes for all years for a certain parameter
    frame.to_csv(subpath+'\sample'+ext+'.csv')

print('done with merging parameters by year\r\n')

all_files = glob.glob(subpath + "/*.csv")#gets all csv files in directory, the only csv files shuld be the ones 
#we just made
li = []
keepcols=['State Code', 'Date', 'AQI', 'Arithmetic Mean', 'Date Local']#desired columns

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    for col in df.columns:
        if(col not in keepcols):#removes unwanted columns. In this case should just be County Code removed
            df.drop(col, axis=1, inplace=True)
    #Changes names of columns based on the unique parameters
    if('WIND' in filename):
        df.rename(columns={'Arithmetic Mean':'Arithmetic Mean (Knots)'}, inplace=True)
    if('TEMP' in filename):
        df.rename(columns={'Arithmetic Mean':'Arithmetic Mean (Degrees Fahrenheit)'}, inplace=True)
    if('PRESS' in filename):
        df.rename(columns={'Arithmetic Mean':'Arithmetic Mean (Millibars)'}, inplace=True)
    if('RH' in filename):
        df.rename(columns={'Arithmetic Mean':'Arithmetic Mean (Percent relative humidity)'}, inplace=True)
    #remove rows of duplicate state and date. We only want 1 row for each state/date. Do not want data by County
    df = df.drop_duplicates(subset=['State Code', 'Date'], keep='first')
    li.append(df)#save each parameter dataframe for later

print('begin intersections\r\n')
frame=li[0]
for i in range(len(li)-1):#intersect each dataframe, only resulting with the rows that are full
    print(i,'\r\n')
    frame=pd.merge(frame,li[i+1],how='inner',on=['State Code', 'Date'])
#remove any other duplicate rows of same State Code and Date
frame = frame.drop_duplicates(subset=['State Code', 'Date'], keep='first')
frame.to_csv(subpath+'\\result'+'\sample.csv')#save file
print('done merging files by parameter')


filename = r'dataForLSTMmodel\result\sample.csv'#path to recently saved csv file
total = pd.read_csv(filename, index_col=2, header=0)#makes the date column the index column
states = list(total['State Code'].unique())#collects all unique state codes
total.drop('Unnamed: 0', axis=1, inplace=True)#get rid of old indexing column that the dataframe made
for stateCode in states:#for every state code, make a smaller csv file of data just for that state
    print('state code is ',stateCode, '\r\n')
    df=total.loc[(total['State Code'] == stateCode)]
    df.drop('State Code', axis=1, inplace=True)#remove state code column since files are seperated by state
    df.to_csv(r'dataForLSTMmodel\byState\State'+str(stateCode)+'.csv')

print('done with making seperate state files')

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(1, 5)))#use 50 LSTM neurons, each taking in 5 values, outputting 1 value
model.add(Dense(1))#final layer summing up LSTM values
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#Prefered parameters for adam optimizer
model.compile(loss='mae', optimizer=adam)#model will use Mean Absolute Error


avgRMSE=0#average root mean square error
RMSEdict={}#records for each state's rmse
for state in states:#train and test the model on every state's data
    statenum=str(state)
    print('CURRENT STATE IS ', statenum)
    # load dataset
    dataset = read_csv(r'dataForLSTMmodel\byState\State'+statenum+'.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize parameters
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values
    halfpoint=int(len(values)*1/2)#use half the data for training, other half for testing
    train = values[:halfpoint, :]
    test = values[halfpoint:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, parameters]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # fit network
    history = model.fit(train_X, train_y, epochs=25, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse, '\r\n')
    avgRMSE+=rmse
    RMSEdict[statenum]=rmse

avgRMSE=float(avgRMSE/len(states))#takes total accumulated rmse and divides by num of states
print('Avg RMSE is ',avgRMSE)
print('records of rmse\r\n', RMSEdict)


# In[ ]:




