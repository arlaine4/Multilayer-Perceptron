import pandas as pd
import numpy as np
from . import maths
import math
import random
from sklearn import preprocessing

class   NeuralNetwork():
    def __init__(self, inputs, nb_layers, nb_hidden_elems, nb_outputs):
        self.input_layer = np.zeros((1, len(inputs)))
        self.input_weights = np.zeros((1, np.size(inputs)))
        self.init_input_layer(inputs, nb_hidden_elems)
        self.weights_ih = np.zeros((len(inputs), nb_hidden_elems))
        #init self.weights_ih
        #init biases

    def init_input_layer(self, inputs, nb_hidden_elems):
        m = inputs.mean(axis=0)
        self.input_layer = np.zeros((1, len(m)))
        mean_all = preprocessing.PowerTransformer()
        mean_all = mean_all.fit(inputs)
        mean_all = mean_all.transform(inputs)
        mean_all = preprocessing.MinMaxScaler(feature_range=(0, 1))
        new_mean_all = mean_all.fit_transform(inputs)
        self.input_layer = new_mean_all[0]
        self.input_weights = xavier_init(np.size(self.input_layer), nb_hidden_elems)

    def __str__(self):
        print("\033[1;3;4mInput Layer \033[0m:\n\n{}\n\n\033[1;3;4mInput Weights\033[0m : \n\n{}\n\n".format(self.input_layer, self.input_weights))

"""class MultilayerPerceptron():
    def __init__(self, inputs, nb_hidden_layers, nb_hidden_elems, nb_outputs):
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_hidden_elems = nb_hidden_elems
        self.input_weights = np.zeros((1, np.size(inputs)))
        self.init_input_layer(inputs)

        self.hidden_layers = np.zeros((self.nb_hidden_layers, self.nb_hidden_elems))

        self.bias = [1 for x in range(self.nb_hidden_layers + 1)]
        self.epoch = 25000
        self.loss = 0.0

        self.nb_outputs = nb_outputs

    def run(self):
        self.hidden_layers[0] = self.create_hidden_layer(self.input_weights, self.bias[0])
        #self.hidden_z[0] = self.get_neuron_output()
    
    def create_hidden_layer(self, weights, bias):
        new_layer = np.zeros((1, self.nb_hidden_elems))
        for i in range(len(new_layer[0])):
            new_layer[0][i] = self.compute_previous_weights(i, weights)
        return new_layer

    def compute_previous_weights(self, i, W):
        computed = (W[0][i] * i) + (W[0][i + 10] * (i + 10)) + (W[0][i + 20] * (i + 20))
        return computed

    def init_input_layer(self, inputs):
        m = inputs.mean(axis=0) #tmp axis assignation for input layer numpy array creation
        self.input_layer = np.zeros((1, len(m)))

        #------------------PowerTransform yeo-johnson------------------------#

        #mean_all = preprocessing.PowerTransformer()
        #mean_all = mean_all.fit(inputs)
        #mean_all = mean_all.transform(inputs)

        #--------------------------------------------------------------------#

        mean_all = preprocessing.MinMaxScaler(feature_range = (0, 1)) #declaration of MIN MAX scaler
        new_mean_all = mean_all.fit_transform(inputs) #scaled features

        self.input_layer = new_mean_all[0]
        self.input_weights = xavier_init(np.size(self.input_layer), self.nb_hidden_elems)

    def __str__(self):
        print("\033[1;3;4mInput layer\033[0m :\n\n{}\n\n\033[1;3;4mBiases\033[0m :\n\n{}\n\n\033[1;3;4mInput weights\033[0m : \n\n{} \
                \n\n\033[1;3;4mFirst Hidden layer\033[0m : \n\n{}\n\n" \
                .format(self.input_layer, self.bias, self.input_weights, self.hidden_layers[0]))"""

#----------------------------------------------------------------#
#                   Activation Functions                         #

def sigmoid(index_e, w_sum, bias):
    e = 2.71828
    z = (b + w_sum) * -1
    sig = (1 / (1 + e**z))
    return sig

def ReLU(z):
    return 0 if z < 0 else 1

#                                                                #
#----------------------------------------------------------------#

def weighted_sum(weights, bias):
    sigma = 0
    for i in range(len(weights - 1)):
        sigma = sigma + (weights[i] * i)
    sigma += bias
    return sigma

def xavier_init(size_layer, size_previous_layer):
    W = np.random.randn(1, size_layer) * np.sqrt(2 / size_previous_layer + size_layer)
    #W = np.random.randn((size_layer), size_previous_layer) * (np.sqrt(2 / (size_previous_layer + size_layer)))
    return W
