import numpy as np

import os

import theano
import theano.tensor as T
import lasagne

'''
Image arrays have the shape (N, 3, 32, 32), where N is the size of the
corresponding set. This is the format used by Lasagne/Theano. To visualize the
images, you need to change the axis order, which can be done by calling
np.rollaxis(image_array[n, :, :, :], 0, start=3).

Each image has an associated 40-dimensional attribute vector. The names of the
attributes are stored in self.attr_names.
'''

data_path = "/home/lmb/Celeb_data"

class Network:
    
    def __init__(self):
        self.network = None
        self.train_images = None
        self.batch_size = None

    def load_data(self):
        self.train_images = np.float32(np.load(os.path.join(
                data_path, "train_images_32.npy"))) / 255.0
        self.train_labels = np.uint8(np.load(os.path.join(
                data_path, "train_labels_32.npy")))
        self.val_images = np.float32(np.load(os.path.join(
                data_path, "val_images_32.npy"))) / 255.0
        self.val_labels = np.uint8(np.load(os.path.join(
                data_path, "val_labels_32.npy")))
        self.test_images = np.float32(np.load(os.path.join(
                data_path, "test_images_32.npy"))) / 255.0
        self.test_labels = np.uint8(np.load(os.path.join(
                data_path, "test_labels_32.npy")))
        
        with open(os.path.join(data_path, "attr_names.txt")) as f:
            self.attr_names = f.readlines()[0].split()
        
        self.batch_size = 200
    
    def build_network(self, input_data):

		self.network = lasagne.layers.InputLayer(shape = (None, 3, 32, 32), input_var = input_data)
		self.network = lasagne.layers.Conv2DLayer(network, num_filters = 32, filter_size = (5, 5), nonlinearity = lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())
		self.network = lasagne.layers.MaxPool2DLayer(network, pool_size = (2, 2))
		self.network = lasagne.layers.Conv2DLayer(network, num_filters = 2, filter_size = (5, 5), nonlinearity = lasagne.nonlinearities.rectify)
		self.network = lasagne.layers.MaxPool2DLayer(network, pool_size = (2, 2))
		self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p = .5),num_units = 256,nonlinearity = lasagne.nonlinearities.rectify)
		self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p = .5),num_units = 10,nonlinearity = lasagne.nonlinearities.softmax)
		
        return self.network
	
net = Network()
net.load_data()

input_data = T.tensor4('inputs')
labels = T.ivector('labels')

build_network(input_data)

prediction = lasagne.layers.get_output(network)

loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable = True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = 0.01, momentum = 0.9)

test_prediction = lasagne.layers.get_output(network, deterministic = True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis = 1), target_var), dtype = theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))