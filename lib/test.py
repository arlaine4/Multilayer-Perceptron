import numpy as np
from sklearn import preprocessing

class NeuralNetwork():
    def __init__(self, inputs, desired_outputs, nb_layers, elems_by_layer):
        self.nb_hidden = [nb_layers, elems_by_layer]
        self.desired_outputs = desired_outputs
        self.activations = []
        self.biases = []
        self.weights = []
        self.backprop_weights = []
        self.lr = 0.001
        self.epoch = 10
        self.err = []

        self.init_lists(inputs)

    def init_lists(self, inputs):
        m = inputs.mean(axis=0)
        self.input_a = np.zeros((1, np.size(m)))
        mean_all = preprocessing.PowerTransformer()
        mean_all = mean_all.fit(inputs)
        mean_all = mean_all.transform(inputs)
        mean_all = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        new_mean_all = mean_all.fit_transform(inputs)

        self.input_a = new_mean_all[0]
        #self.input_a = self.input_a.reshape(30, 1)
        self.activations = self.input_a
        self.add_bias_index(np.size(self.input_a))
        for i in range(self.nb_hidden[1]):
            self.weights.append(self.xavier_init(np.size(self.input_a), self.nb_hidden[1]))
            self.backprop_weights.append(self.xavier_init(np.size(self.input_a), self.nb_hidden[1], True))

    def feedforward(self):
        pass

    def backpropagation(self):
        pass

    def add_bias_index(self, size):
        self.biases.append([1 for i in range(size)])

    def __str__(self):
        print("\033[1;3;4mActivation \033[0m: \n\n{}\n\n".format(self.activations))
        print("\033[1;3;4mWeights\033[0m:\n\n{}\n\n".format(self.weights))
        print("\033[1;3;4mBiases\033[0m:\n\n{}\n\n".format(self.biases))

    def xavier_init(self, size, size_previous, init=False):
        if init is False:
            W = np.random.randn(1, size) * np.sqrt(2 / size_previous + size)
        elif init:
            array = [0.0 for k in range(size_previous)]
            W = [array for i in range(size)]
            return W[0]
        return list(W[0])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def der_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
