import os
import numpy as np

import tensorflow
#from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

def load_mnist_data():
    path = os.path.split(__file__)[0]
    labels_path = os.path.join(path,'data','train-label-onehot.npy')
    images_path = os.path.join(path,'data','train-image.npy')

    labels = np.load(labels_path)
    images = np.load(images_path)
    return labels,images

# build model
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
model = Sequential()

labels,images = load_mnist_data()
images = images/255
images=images.squeeze()
offset = 40000
x_train = images[0:offset,:]
y_train = labels[0:offset,:]
x_test = images[offset:,:]
y_test = labels[offset:,:]

print('deep dive!')


# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
input_shape=(img_rows*img_cols,)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Activation(128, input_shape=input_shape))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

from tensorflow.keras.optimizers import RMSprop

opt = RMSprop(lr=0.0001, decay=1e-6)

print("compiling")
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=opt, #tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
