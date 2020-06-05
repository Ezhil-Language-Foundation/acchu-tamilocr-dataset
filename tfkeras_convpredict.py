from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import tamil
import copy
import glob
from matplotlib import pyplot as plt
from time import sleep
# Adapted from: https://keras.io/examples/mnist_cnn/

# 1) Setup letters to be built
uyir_plus_ayutham = copy.copy(tamil.utf8.uyir_letters)
uyir_plus_ayutham.append( tamil.utf8.ayudha_letter )

def to_tamil_letter(idx):
    return(uyir_plus_ayutham[idx])

def load_acchu_data(mode='train'):
    path = os.path.split(__file__)[0]
    labels_path = os.path.join(path,'data',mode+'-label-onehot.npy')
    images_path = os.path.join(path,'data',mode+'-image.npy')
    labels = np.load(labels_path)
    images = np.load(images_path)
    return labels,images

batch_size = 128
num_classes = 13
epochs = 200

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

#model = keras.models.load_model('acchu_conv_model_5')
model = keras.models.load_model('acchu_conv3_model_2')
model.summary()

expected = ['a','aa','e','ee','u','uu','eh','aeh','ai','o','oh','au','ak']
assert len(expected) == 13
stats = 0.0
total = 0.0
msg = []
for letter_image in glob.glob('letters-hand-drawn-corrected/*.npy'):#npy/
    total += 1.0
    print("#"*32)
    data = np.load(letter_image)
    #data = data.transpose()
    plt.imshow(data)
    plt.show()
    #sleep(5)
    data = data.reshape(1,28,28,1)/255.0
    output = model.predict(data)
    print(letter_image)
    predicted = np.argmax(output[0])
    print("predicted class=>",to_tamil_letter(predicted))
    sfx = letter_image.split('_')[1].replace('.npy','')
    print(sfx)
    if predicted != expected.index(sfx):
        stats += 1.0
        msg.append("{0} misclassified as {1}".format(to_tamil_letter(expected.index(sfx)),to_tamil_letter(predicted)))
    print(output[0])
print("Failed cases: %g = (%g/%g)"%(stats/total,stats,total))
print("\n".join(msg))
if False:
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
