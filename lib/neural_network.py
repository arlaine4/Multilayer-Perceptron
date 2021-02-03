import pandas as pd
import numpy as np
from . import maths
import math
import random
from sklearn import preprocessing

np.seterr(divide = 'ignore')

class   NeuralNetwork():
    def __init__(self, inputs, nb_layers, nb_hidden_elems, nb_outputs, desired_outputs):
        self.all_desired_outputs = np.array(desired_outputs) #desired 1 and 0 for each cell that i'm going to feed
        self.input_layer = np.zeros((np.size(inputs), 1)) #input layer
        self.input_weights = np.zeros((np.size(inputs), 1)) #input weight
        self.init_input_layer(inputs, nb_hidden_elems) 
        self.weights_ih = np.zeros((nb_hidden_elems, np.size(self.input_layer))) #input to 1st hidden weight matrix
        self.bias_h = np.ones((nb_hidden_elems, 1), dtype=np.int16) #bias matrix
        self.hidden_layers = np.zeros((nb_layers, nb_hidden_elems)) #matrix of hidden layers values (z) then sigmoid(z)

    def init_input_layer(self, inputs, nb_hidden_elems):
        """Init input layer with inputs value, scale the date with yeo-johnson
        PowerTransform, then using xavier_init method to give input layer weights"""
        m = inputs.mean(axis=0)
        self.input_layer = np.zeros((1, np.size(m)))
        #---------Input scaling with PowerTransform yeo-johnson--------------#
        mean_all = preprocessing.PowerTransformer()
        mean_all = mean_all.fit(inputs)
        mean_all = mean_all.transform(inputs)
        mean_all = preprocessing.MinMaxScaler(feature_range=(0, 1))
        new_mean_all = mean_all.fit_transform(inputs)
        #--------------------------------------------------------------------#
        self.input_layer = new_mean_all[0]
        self.input_layer = self.input_layer.reshape(31, 1)
        self.input_weights = xavier_init(np.size(self.input_layer), nb_hidden_elems)

    def init_weights_ih(self):
        """Matrix product between input layer and input_weights to get the 
        input to hidden weight matrix"""
        self.weights_ih = self.input_layer @ self.input_weights
        print(self.weights_ih.shape)

    def run(self):
        for epoch in range(20000):
            input_hi
        self.init_weights_ih()

    def __str__(self):
        print("\033[1;3;4mInput Layer\033[0m :\n\n{}\n\n\033[1;3;4mInput Weights\033[0m : \n\n{}\n\n".format(self.input_layer, self.input_weights))
        print("\033[1;3;4mBias\033[0m : \n\n{}\n\n".format(self.bias_h.T))
        print("\033[1;3;4mHidden Layers\033[0m :\n\n{}\n\n".format(self.hidden_layers.T))


#----------------------------------------------------------------#
#                   Utils Functions for every NN                 #

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(z):
    return 0 if z < 0 else 1

def weighted_sum(weights, bias):
    sigma = 0
    for i in range(1, len(weights)):
        sigma = sigma + (weights[i] * i)
    sigma += bias
    return sigma

def xavier_init(size_layer, size_previous_layer):
    W = np.random.randn(1, size_layer) * np.sqrt(2 / size_previous_layer + size_layer)
    #W = np.random.randn((size_layer), size_previous_layer) * (np.sqrt(2 / (size_previous_layer + size_layer)))
    return W

#                                                                #
#----------------------------------------------------------------#
