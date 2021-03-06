# %load mlp.py

"""
mlp.py

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class MLP(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            #a = relu(np.dot(w, a)+b)
            #a = tanh(np.dot(w, a)+b)
            a = sigmoid(np.dot(w, a)+b)
        return a

    def map_list_to_dict(self, _list):
        return {i:"{:.4f}".format(v) for i,v in enumerate(_list)}

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                precision_per_class = self.evaluate_classes_precision(test_data, 10)
                recall_per_class = self.evaluate_classes_recall(test_data, 10)
                total_acc = self.evaluate_accuracy(test_data)
                print('Epoch: ' + str(j))
                print('Total Accuracy: ' + str(total_acc) + '/' + str(n_test))
                print('Precision per class:')
                print(self.map_list_to_dict(precision_per_class))
                print('Recall per class:')
                print(self.map_list_to_dict(recall_per_class))
                print('-----------------------------------')
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            #activation = relu(z)
            #activation = tanh(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
            #relu_prime(zs[-1])
            #tanh_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            #sp = relu_prime(z)
            #sp = tanh_prime(z)
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def get_output_tuples(self, test_data):
        return [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

    def evaluate_accuracy(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = self.get_output_tuples(test_data)
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_classes_precision(self, test_data, class_number):
        """Calculates and returns precision per class."""
        output_per_class = [0] * class_number
        correct_output_per_class = [0] * class_number

        for (x, y) in test_data:
            a = self.feedforward(x)
            print(a)
        test_results = self.get_output_tuples(test_data)

        for (x, y) in test_results:
            if x == y:
                correct_output_per_class[int(y)] += 1
            output_per_class[int(x)] += 1

        precision_per_class = [0] * class_number
        for i,v in enumerate(correct_output_per_class):
            if output_per_class[i] == 0:
                precision_per_class[i] = 0
            else:
                precision_per_class[i] = v / output_per_class[i]
        
        return precision_per_class

    def evaluate_classes_recall(self, test_data, class_number):
        """Calculates and returns recall per class."""
        correct_output_per_class = [0] * class_number
        total_per_class = [0] * class_number
        
        test_results = self.get_output_tuples(test_data)

        for (x, y) in test_results:
            if x == y:
                correct_output_per_class[int(y)] += 1
            total_per_class[int(y)] += 1
        
        recall_per_class = [0] * class_number
        for i,v in enumerate(correct_output_per_class):
            if total_per_class[i] == 0:
                recall_per_class[i] = 0
            else:
                recall_per_class[i] = v / total_per_class[i]
        
        return recall_per_class

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return (1-tanh(z)**2)

def relu(z):
    return np.maximum(z, 0)

def single_value_relu_prime(z):
    if z > 0:
        return 1
    else:
        return 0

def relu_prime(z):
    a = np.vectorize(single_value_relu_prime)
    return a(z)
