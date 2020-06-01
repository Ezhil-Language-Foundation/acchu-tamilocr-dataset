from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
# Adapted from: https://keras.io/examples/mnist_cnn/

def load_acchu_data(mode='train'):
    path = os.path.split(__file__)[0]
    labels_path = os.path.join(path,'data',mode+'-label-onehot.npy')
    images_path = os.path.join(path,'data',mode+'-image.npy')
    labels = np.load(labels_path)
    images = np.load(images_path)
    return labels,images

batch_size = 128
num_classes = 13
epochs = 50

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
y_train, x_train  = load_acchu_data('train')
y_test, x_test    = load_acchu_data('test')

x_train = x_train.reshape(len(x_train), img_rows, img_cols,1)
x_test = x_test.reshape(len(x_test), img_rows, img_cols,1)
input_shape = (img_rows, img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

if True:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten(input_shape=(64,num_classes)))
    model.add(Dense(num_classes, activation='softmax'))
else:
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Flatten(input_shape=(512, num_classes)),
    keras.layers.Dense(num_classes,activation='softmax')])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('acchu_conv_model')
