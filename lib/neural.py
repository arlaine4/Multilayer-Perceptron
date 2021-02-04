import numpy as np
import pandas as pd
from sklearn import preprocessing

class NeuralNetwork():
    def __init__(self, inputs, desired_ouputs, nb_hidden_layers, nb_hidden_elems, nb_outputs, test_or_train='train'):
        self.input_weights = None
        self.biases = []
        self.nb_hidden = {'nb_layers' : nb_hidden_layers, 'nb_elem' : nb_hidden_elems}
        self.input_data_scaling(inputs)
        self.output_matrix = np.array(desired_ouputs)
        self.nb_ouput_neurons = nb_outputs
        self.epoch = 20000
        self.memory_weights = []
        self.memory_biases = []
        #self.biases = np.array((1, (self.nb_hidden['nb_layers'] * self.nb_hidden['nb_elem']) + self.nb_ouput_neurons + np.size(self.input_a)))
        #self.memory_weights = np.array((1, self.nb_hidden['nb_layers'] + 2))
        #self.memory_biases = np.array((1
    
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
        self.biases = np.ones((1, np.size(self.input_a)))
        for i in range(self.nb_hidden['nb_elem'][0]):
            self.input_weights[i] = xavier_init(np.size(self.input_a), self.nb_hidden['nb_elem'][0])

    def fit(self, mode='train'):
        if mode == 'train':
            #for i in range(self.epoch):
            i = 0
            #next_input = self.input_a @ self.input_weights + self.biases[0]


    
    def __str__(self):
        print("\033[1;3;4mInput activation \033[0m:\n\n{}\n\n".format(self.input_a))
        print("\033[1;3;4mInput Weights \033[0m:\n\n{}\n\n".format(self.input_weights))
        print("\033[1;3;4mDesired outputs \033[0m:\n\n{}\n\n".format(self.output_matrix))

    #---------------------------------------------------#
    #               Activation Functions                #

    def sigmoid(self, x):
        return 1 / 1 + np.exp(x)

    def sigmoid_der(self, x):
        return sigmoid(x) * (1 - sigmoid(x))

    #                                                   #
    #---------------------------------------------------#

def xavier_init(size_layer, size_previous_layer):
    W = np.random.randn(1, size_layer) * np.sqrt(2 / size_previous_layer + size_layer)
    return W

