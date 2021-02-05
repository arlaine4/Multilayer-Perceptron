import numpy as np
np.seterr(divide = 'ignore')
import pandas as pd
from sklearn import preprocessing

class NeuralNetwork():
    def __init__(self, inputs, desired_outputs, nb_hidden_layers, nb_hidden_elems, nb_outputs, test_or_train='train'):
        self.init_matricies(inputs, desired_outputs, nb_hidden_layers, nb_hidden_elems, nb_outputs)
        """self.input_weights = None
        self.nb_hidden = {'nb_layers' : nb_hidden_layers, 'nb_elem' : nb_hidden_elems}
        self.activations = [np.array(1,
        self.weights = [[]]
        self.output_matrix = np.array(desired_ouputs)
        self.nb_ouput_neurons = nb_outputs
        self.epoch = 20000
        self.biases = [[]]
        self.input_data_scaling(inputs)"""
    
    def init_matricies(self, inputs, desired_outputs, nb_hidden_layers, nb_hidden_elems, nb_outputs):
        self.desired_outputs = desired_outputs
        self.nb_hidden = {'nb_layers' : nb_hidden_layers, 'nb_elem' : nb_hidden_elems}
        m = inputs.mean(axis=0)
        self.activations = list(np.ones((1, np.size(m))))
        self.biases = [np.ones((1, len(inputs)))]
        self.epoch = 2000
        self.output_matrix = np.zeros((1, nb_outputs))
        self.weights = [np.zeros((self.nb_hidden['nb_elem'][0], len(inputs)))]
        self.nb_layers = self.nb_hidden['nb_layers'] + 2
        self.input_data_scaling(inputs)

    def append_new_np_array(self, base_array, new_nb_elems):
        new_array = np.zeros((1, new_nb_elems))
        base_array.append(new_array)
        #base_array = np.vstack((base_array, new_array))
        return base_array

    def input_data_scaling(self, data):
        m = data.mean(axis=0)
        self.input_a = np.zeros((1, np.size(m)))
        #---------Input scaling with PowerTransform yeo-johnson--------------#
        mean_all = preprocessing.PowerTransformer()
        mean_all = mean_all.fit(data)
        mean_all = mean_all.transform(data)
        mean_all = preprocessing.MinMaxScaler(feature_range=(0, 1))
        new_mean_all = mean_all.fit_transform(data)
        #--------------------------------------------------------------------#
        self.input_a = new_mean_all[0]
        self.input_a = self.input_a.reshape(30, 1)
        self.input_weights = np.zeros((self.nb_hidden['nb_elem'][0], np.size(self.input_a)))
        self.weights = self.input_weights
        self.biases = np.ones((1, np.size(self.input_a)))
        self.activations[0] = self.input_a
        for i in range(self.nb_hidden['nb_elem'][0]):
            self.input_weights[i] = xavier_init(np.size(self.input_a), self.nb_hidden['nb_elem'][0])

    def fit(self, data, mode='train'):
        self.activations = self.append_new_np_array(self.activations, 20)
        if mode == 'train':
            #for i in range(self.epoch):
            k = 1
            while k < self.nb_hidden['nb_layers']:
                for l in range(self.nb_hidden['nb_elem'][k]):
                    self.activations[k][l] = weighted_sum(self.input_weights[k - 1], self.activations[k - 1], self.biases[k - 1])
                    ### PROB HERE
                    for elem in range(len(self.activations[k - 1])):
                        self.activations[k][0][0] = self.sigmoid(self.activations[k - 1])
                k += 1
                
    def __str__(self):
        print("\033[1;3;4mActivation input\033[0m : \n\n", self.activations[0])
        print("\033[1;3;4mBiases\033[0m : \n\n", self.biases)
        print("\033[1;3;4mWeights\033[0m : \n\n", self.weights)

    #---------------------------------------------------#
    #               Activation Functions                #

    def sigmoid(self, x):
        return 1 / 1 + np.exp(x)

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    #                                                   #
    #---------------------------------------------------#

def weighted_sum(weights, activation, biases):
    sum_ = 0
    print("W : ", weights[0])
    print("A : ",activation[0][0])
    print("B : ",biases[0])
    print(len(weights), len(activation), len(biases))
    for i in range(len(weights)):
        sum_ += (weights[i] * activation[i][0]) + biases[i]
    return sum_

def xavier_init(size_layer, size_previous_layer):
    W = np.random.randn(1, size_layer) * np.sqrt(2 / size_previous_layer + size_layer)
    return W

