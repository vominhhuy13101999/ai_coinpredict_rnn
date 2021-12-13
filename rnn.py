import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from statistics import mean,stdev,pstdev
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
def day_month(data,name):
  open=[data.Open[0]]
  close=[]
  date=[data.Date[0]]
  for i in range(len(data)):
    if data.Date[i].is_month_start:
      open.append(data.Open[i])
      date.append(data.Date[i])
    elif data.Date[i].is_month_end:
      close.append(data.Close[i])


    
  close.append(0)
  dic={"Date":date,"Open":open,"Close":close}
  dic=pd.DataFrame.from_dict(dic)
  dic.to_excel("{name}_monthly.xlsx")
  return dic
def day_week(data,name):
  open=[data.Open[0]]
  close=[]
  date=[data.Date[0]]
  for i in range(len(data)):
    if data.Date[i].dayofweek==0:
      open.append(data.Open[i])
      date.append(data.Date[i])
    elif data.Date[i].dayofweek==6:
      close.append(data.Close[i])


    
  close.append(0)
  dic={"Date":date,"Open":open,"Close":close}
  dic=pd.DataFrame.from_dict(dic)
  dic.to_excel("{}_weekly.xlsx".format(name))
  return dic
class return_processing():
  def __init__(self):
    # self.data=data
    pass
  def fit(self,data):
    self.data=data.to_numpy()
    self.fit_data=(data.shift(-1)/data-1).shift(1)
  def transform(self,data):
    try:  
      return self.fit_data
    except:
      print("data not fit yet")
      raise ValueError
  def fit_transform(self,data):
    self.fit(data)
    return self.transform(data)
  def inverse(self,data,head=True,start=0):
    if head:
      try:
        n=len(data)
        return((1+data)*self.data[:n])
      except:
        print("data not fit yet")
        raise ValueError
    else:
      if start==0:
        try:
          n=len(data)
          return ((1+data)*self.data[-n-1:-1])
        except:
          print("data not fit yet")
          raise ValueError
      else:
        try:
          n=len(data)
          return ((1+data)*self.data[start-1:start+n-1])#.drop(start-1)
        except:
          print("data not fit yet")
          raise ValueError
def create_dataset(dataset1, look_back,length,test=False):

    dataX, dataY = [], []
    if not test:
      # for i in range(length-look_back+1):
      for i in range(length-look_back):

          a = dataset1.Open_z[i:(i+look_back)+1]
          dataX.append(a)
          dataY.append(dataset1.Close_z[i + look_back])
      return np.array(dataX), np.array(dataY)
    else:
      start=length-look_back
      for i in range(start,len(dataset1)-look_back):
          a = dataset1.Open_z[i:(i+look_back)+1]
          dataX.append(a)
          dataY.append(dataset1.Close_z[i + look_back])
      # a = dataset[len(dataset)-look_back:, 0]
      # dataX.append(a)
      # dataY.append(0)

      return np.array(dataX), np.array(dataY)