import tensorflow as tf
import os
import numpy as np
# load the saved model from 'tfdemo.py' and run some data.

def load_acchu_data(mode='train'):
    path = os.path.split(__file__)[0]
    labels_path = os.path.join(path,'data',mode+'-label-onehot.npy')
    images_path = os.path.join(path,'data',mode+'-image.npy')
    labels = np.load(labels_path)
    images = np.load(images_path)
    return labels,images

# model parameters
batch_size = 128
num_classes = 13
epochs = 12
img_rows, img_cols = 28, 28# input image dimensions

# laod test/validation data:
test_labels,test_images = load_acchu_data('test')
offset=0
x_test = test_images[offset:,:]
y_test = test_labels[offset:,:]

filename = os.path.abspath(__file__)
basedir =  os.path.split(filename)[0]
model_name = 'tamil_model_ckpt'
model_path = os.path.join(basedir,'tamil_model_ckpt',model_name)

tf.reset_default_graph()

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 13 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#saver= tf.train.Saver()
#saver.restore(sess, model_path)
#all_vars = tf.get_collection('vars')
#for v in all_vars:
#    v_ = sess.run(v)
#    print(v_)

#x = sess.graph.get_tensor_by_name("x")
#y = sess.graph.get_tensor_by_name("y")
y0 = sess.run(Y,feed_dict={X:x_test[1,:].reshape(1,784)})
print(y_test[1,:])
print(y0)
