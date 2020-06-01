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
epochs = 24*3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
y_train, x_train  = load_acchu_data('train')
y_test, x_test    = load_acchu_data('test')

x_train = x_train.reshape(len(x_train), img_rows, img_cols)
x_test = x_test.reshape(len(x_test), img_rows, img_cols)
input_shape = (img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = keras.models.load_model('acchu_model_3')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('acchu_model_4')
