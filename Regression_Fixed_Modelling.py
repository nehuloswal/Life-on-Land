"""Team Members: Nehul Oswal, Fedrick Durao, Anirban Bhattacharya
This code is used to calculate the correlation between pollutant like SO2, CO, NO2 and OZONE to weather feature like wind, temperature,
pressure and RH for each state. In order to calculate the correlation values we used multiple regression while modelling county as a fixed 
effect and including dummy variables for year and month in order to account for seasonality. We also tried mixed modelling by modelling 
county as random effect with respect to the weather features.
Dataframe: Spark used for combining data and for running regression for each state parallely
Cnncept: Mutiple Regression with fixed and mixed modelling (Place: Regression_fixed_modelling(), Regression_mixed_modelling())
Machine: Google Cloud Dataproc running Ubuntu"""


#All the necessary packages
from pyspark import SparkContext,SparkConf
import json
import sys
import re
import csv
import numpy as np
import pandas as pd
from scipy import stats
from operator import add
from functools import reduce
from pyspark.sql import SQLContext, SparkSession

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()


#Function for multiple regression with county as a random effect w.r.t weather features
def Regression_mixed_modelling(x, hypothesis_count):
    x_df_with_dummy = pd.DataFrame(x)
    x_df_with_dummy.columns = ['Date','County','Pollutant','Wind','Temp','Pressure','RH']
    x_df_with_dummy.Date = pd.to_datetime(x_df_with_dummy.Date)
    x_df_with_dummy = x_df_with_dummy.sort_values(by = 'Date') #Sorting the records by date in ascending order
    L = ['year','month']
    date_gen = (getattr(x_df_with_dummy['Date'].dt, i).rename(i) for i in L) #Adding year and month columns to the dataframe
    x_df_with_dummy = x_df_with_dummy.join(pd.concat(date_gen, axis=1))
    x_df_with_dummy = x_df_with_dummy.drop(['Date'], axis = 1)
    columns_to_normalize = ['Pollutant', 'Wind','Temp','Pressure','RH']  #Normalizing the columns
    x_df_with_dummy[columns_to_normalize] = x_df_with_dummy[columns_to_normalize].apply(lambda x: (x - x.mean())/x.std())
    x_df = x_df_with_dummy
    endog = x_df_with_dummy['Pollutant']   #Dependent variable
    exog = x_df_with_dummy[['Wind','Temp','Pressure','RH','year','month']] #Independent variable
    mixed = sm.MixedLM(endog, exog, x_df_with_dummy["County"], exog[['Wind','Temp','Pressure','RH']]).fit()  #Modelling county as a random effect
    w = np.array(mixed.params)
    w = w[0:6,]  #Getting the weights for 'Wind','Temp','Pressure','RH','year','month'
    endog = endog.to_numpy()
    exog = exog.to_numpy()
    endog = np.reshape(endog, (len(endog),1))
    exog = np.reshape(exog, (len(exog),len(exog[0])))
    
    df = float(len(x) - (len(x_df_with_dummy.columns) - 1))
    rss = np.sum(np.power((endog - np.dot(exog,w)),2))
    s_squared = rss/df
    p_vals = []
    for i in range(0,4):  #Calculating the pval for wind, temp, pressure, RH
        se = np.sum(np.power((exog[:,i]),2))
        t_val = (w[i]/np.sqrt(s_squared/se))
        pval = float(stats.t.sf(np.abs(t_val), df))*2
        p_vals.append(pval*hypothesis_count)

    w = np.round(w[0:4],4).tolist()  #Getting the weights for wind, temp, pressure, RH
    return (w, p_vals)



#Function for multiple regression with county as a fixed effect
def Regression_fixed_modelling(x, hypothesis_count):
    x_df = pd.DataFrame(x)
    x_df.columns = ['Date','County','Pollutant','Wind','Temp','Pressure','RH']
    x_df_with_dummy = pd.concat([x_df,pd.get_dummies(x_df.County, drop_first = True)],axis=1) #Adding the dummy variable county
    x_df_with_dummy.Date = pd.to_datetime(x_df_with_dummy.Date)
    x_df_with_dummy = x_df_with_dummy.sort_values(by = 'Date') #Sorting the records by date in ascending order
    L = ['year','month']
    date_gen = (getattr(x_df_with_dummy['Date'].dt, i).rename(i) for i in L) #Adding year and month columns to the dataframe
    x_df_with_dummy = x_df_with_dummy.join(pd.concat(date_gen, axis=1))
    x_df_with_dummy = pd.concat([x_df_with_dummy,pd.get_dummies(x_df_with_dummy.year)],axis=1) #Adding the dummy variable for year
    x_df_with_dummy = pd.concat([x_df_with_dummy,pd.get_dummies(x_df_with_dummy.month)],axis=1) #Adding the dummy variable month
    x_df_with_dummy = x_df_with_dummy.drop(['Date','County'], axis = 1)
    columns_to_normalize = ['Pollutant', 'Wind','Temp','Pressure','RH']  #Normalizing the columns
    x_df_with_dummy[columns_to_normalize] = x_df_with_dummy[columns_to_normalize].apply(lambda x: (x - x.mean())/x.std())
    dependent = x_df_with_dummy.iloc[:,0:1].values.tolist() #Getting the dependent variables i.e. pollutant
    dependent = reduce(add, dependent)
    independent = x_df_with_dummy.iloc[:,1:].values.tolist() #Getting the independent variable i.e. wind, temp etc

    dependent = np.reshape(dependent, (len(dependent),1))
    independent = np.reshape(independent, (len(independent),len(independent[0])))

    independent = np.hstack((np.ones((len(independent),1)), independent))  #Adding the ones column
    independent_inv = np.linalg.pinv(independent)  #Calculating Pseudo-inverse of independent variables
    w = np.dot(independent_inv,dependent)  #Calculating weights
    
    df = float(len(x) - (len(x_df_with_dummy.columns) - 1))
    rss = np.sum(np.power((dependent - np.dot(independent,w)),2))
    s_squared = rss/df
    p_vals = []
    for i in range(1,5): #Calculating the pval for wind, temp, pressure, RH
        se = np.sum(np.power((independent[:,i]),2))
        t_val = (w[i,0]/np.sqrt(s_squared/se))
        pval = float(stats.t.sf(np.abs(t_val), df))*2
        p_vals.append(pval*hypothesis_count)

    w = np.round(w[1:5,:],4).tolist() #Getting the weights for wind, temp, pressure, RH
    return (w,p_vals)
    

#Reading the pollutant and weather feature files
pollutant = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\Courses\CSE 545\Project\pollutant_small.csv')
wind = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\Courses\CSE 545\Project\Winds_small.csv')
temp = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\Courses\CSE 545\Project\emp_small.csv')
pressure = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\Courses\CSE 545\Project\press_small.csv')
RH = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\Courses\CSE 545\Project\dew_small.csv')

#Combining all the files based on the conditions of same state code, county code and date 
conditions = ['State Code', 'County Code', 'Date Local']
#Inner joining the pollutant records and wind records 
df1 = pollutant.join(wind, on = conditions, how = 'inner').select(pollutant['State Code'],pollutant['Date Local'],pollutant['County Code'], pollutant['Pollutant'], wind['Wind'])
#Inner joinung  df1 and temp records
df2 = df1.join(temp, on = conditions, how = 'inner').select(df1['State Code'],df1['Date Local'],df1['County Code'], df1['Pollutant'],df1['Wind'],temp['Temperature'])
#Inner joinung  df1 and Pressure records
df3 = df2.join(pressure, on = conditions, how = 'inner').select(df2['State Code'],df2['Date Local'],df2['County Code'], df2['Pollutant'],df2['Wind'],df2['Temperature'], pressure['Pressure'])
#Inner joinung  df1 and RH records
df4 = df3.join(RH, on = conditions, how = 'inner').select(df3['State Code'],df3['Date Local'],df3['County Code'], df3['Pollutant'],df3['Wind'],df3['Temperature'], df3['Pressure'], RH['RH'])
input = df4.rdd.map(tuple) #COnverting the obtained dataframe to RDD
#Making state code as the key and group all the records by the state code so that the regression can be run in parallel
state_key = input.map(lambda line: (line[0],[[line[1],line[2],float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7])]])).reduceByKey(lambda a,b : a + b)
#Getting the count of states i.e. no of hypothesis
hypothesis_count = state_key.count()
#Using regression with county as a fixed effect
correlation_fixed = state_key.mapValues(lambda x: Regression_fixed_modelling(x, hypothesis_count)).collect()
#Using regression with county as a random effect w.r.t temperature, wind, pressure and RH.
correlation_mixed = state_key.mapValues(lambda x: Regression_mixed_modelling(x, hypothesis_count)).collect()
print(*correlation_fixed, sep = '\n')
print(*correlation_mixed, sep = '\n')