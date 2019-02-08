import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import pandas as pd
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense  
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


raw_data = np.loadtxt('./challenge/poker-hand-training-true.data.txt', delimiter=',')

data = np.array(raw_data).astype('float32')

print(data.shape)

x_train = data[:,:10]
y_train = data[:,10]


scaler = MinMaxScaler()

X_train = scaler.fit_transform(x_train)
X_train = X_train.reshape(len(X_train),-1)
Y_TrainOne_Hot = np_utils.to_categorical(y_train)

print(X_train.shape)
print(y_train.shape)
print(Y_TrainOne_Hot.shape)


model = Sequential()

model.add(Dense(units=100,
                input_dim=10,
                kernel_initializer='normal',
                activation='relu'))

model.add(Dropout(0.2))


model.add(Dense(units=500,
                kernel_initializer='normal',
                activation='relu'))

model.add(Dense(units=500,
                kernel_initializer='normal',
                activation='relu'))


model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
print(model.summary())


model.compile(loss='categorical_crossentropy',
                optimizer='adam',metrics=['accuracy'])

checkpoint = ModelCheckpoint('./lvinno.h5', monitor='acc', 
                             verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

train_history = model.fit(x=X_train,
                          y=Y_TrainOne_Hot,
                          epochs=200,
			  validation_split=0.2,
                          batch_size=200,
                          verbose=2,callbacks = callbacks_list)


                        
