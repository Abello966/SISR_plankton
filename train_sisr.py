import numpy as np
import keras as kr
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

def DongCNN(in_shape):
    model = Sequential()
    
    model.add(Conv2D(64, (9, 9), padding='same', input_shape=in_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(1, (5,5), padding='same'))
    
    return model

def AlexNetCNN(in_shape):

    alpha = 0.3

    model = Sequential()
    model.add(Conv2D(32, (11, 11), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(32, (5, 5), padding='same')) 
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (3, 3), padding='same')) 
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))


    model.add(Conv2D(1, (3, 3), padding='same'))
    return model






#Carrega Dados
from sklearn.model_selection import train_test_split
dataset = np.load("dataset_black.npy")
X = dataset[:,1,:,:]
Y = dataset[:,0,:,:]
X = X.reshape(X.shape + (1,))
Y = Y.reshape(Y.shape + (1,))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, train_size=0.9)

model = AlexNetCNN(X_train[0].shape)
opt = Adam()
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
checkpoint = ModelCheckpoint("CNNSISR_black", monitor='val_loss', verbose=0, save_best_only=True)
callback_list = [early_stop, checkpoint]

model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1, callbacks=callback_list, validation_data=(X_test, y_test))
