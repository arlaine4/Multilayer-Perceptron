import pandas as pd
import numpy as np
from copy import deepcopy ##
from . import math
from sklearn import preprocessing

class MultilayerPerceptron():
    def __init__(self, inputs, nb_hidden_layers, nb_outputs):
        self.nb_hidden_layers = nb_hidden_layers
        self.init_input_layer(inputs)
        self.bias = 0
        self.init_bias(nb_hidden_layers + 1)

    def init_bias(self, nb_layers):
        self.bias = [1 for x in range(nb_layers)]

    def init_input_layer(self, inputs):
        m = inputs.mean(axis=0) #tmp axis assignation for input layer numpy array creation
        self.input_layer = np.zeros((1, len(m)))
        mean_all = preprocessing.MinMaxScaler(feature_range = (0, 1)) #declaration of MIN MAX sclaer
        new_mean_all = mean_all.fit_transform(inputs) #scaled features
        self.input_layer = new_mean_all[0]
    
    def __str__(self):
        print("Input layer :\n\n{}\n\nBiases :\n\n{}".format(self.input_layer, self.bias))

#----------------------------------------------------------------#
#                   Activation Functions                         #

def sigmoid(index_e, w_sum, bias):
    e = 2.71828
    z = (b + w_sum) * -1
    sig = (1 / (1 + e**z))
    return sig

def ReLU(z):
    return 0 if z < 0 else z
    #return 0 if z < 0 else return 1

#                                                                #
#----------------------------------------------------------------#

def create_multilayer_perceptron(nb_hidden_layers, inputs, size_output_layer):
    pass

def create_perceptron(inputs, weights, activation_func, bias):
    bias = 1    

def xavier_init(layer, previous_layer):
    W = np.random.randn((np.size(layer), np.size(previous_layer)) * (np.sqrt(2 / (np.size(previous_layer) + np.size(layer)))))
    return W

def weighted_sum(weights, bias):
    weighted_sum = 0
    for i in range(len(weights)):
        weighted_sum += weights[i] * i
    weighted_sum += 1 * bias
    return weighted_sum
