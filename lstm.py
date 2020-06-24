import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from sklearn import preprocessing

csvroot="./data5.csv"
timeWindow=100
nextNum=3
dataNum=200


df= pd.read_csv(csvroot,sep=',',header = None)
df=df.reindex(index=df.index[::-1])
data=[]
for tup in zip(df[1],df[2],df[3],df[4],df[5],df[6]):
    data.append(tup)
data.pop()
data=np.array(data,dtype='float64')
sc = MinMaxScaler(feature_range=(0, 1))
data = sc.fit_transform(data)


train_X, train_Y = [], []
for i in range(dataNum):
    a = data[i:(i+timeWindow),:]
    train_X.append(a)        
    b = data[(i+timeWindow):(i+timeWindow+nextNum),0]
    train_Y.append(b)
trainX = np.array(train_X,dtype='float64')
trainY = np.array(train_Y,dtype='float64')


model = Sequential()
model.add(LSTM(128,input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))
model.add(Activation('relu'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


model.fit(trainX,trainY,batch_size=1,epochs=50)
score = model.evaluate(trainX, trainY, batch_size=10)