# -*- coding: utf-8 -*-
"""
Created on Sat May 19 10:31:29 2018

@author: Kawin-PC
"""

#import Keras library
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

#initial value and directory
dir_train='data/train/'
dir_valid='data/validation/'
img_size=150
batch_size=32
epochs=5
input_shape=(img_size,img_size,3)

#part 1 Create model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#part 2 fit model to image
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
valid_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory(dir_train,target_size=(img_size,img_size),batch_size=batch_size,class_mode='binary')
valid_set=valid_datagen.flow_from_directory(dir_valid,target_size=(img_size,img_size),batch_size=batch_size,class_mode='binary')
model.fit_generator(train_set,epochs=epochs,verbose=1,validation_data=valid_set)
model.save_weights("model_softmax.h5")
model.save("model_softmax.h5")