import math
import numpy as np

def main_neural(train, desired_outputs, epoch):
    net = Network([30, 20, 20, 1])
    Neuron.eta = 0.1
    Neuron.alpha = 0.1
    for i in range(epoch):
        err = 0
        for j in range(len(train)):
            net.setInput(train[j])
            net.feedForward()
            net.backPropagate(desired_outputs[j])
            err = err + net.getError(desired_outputs[j])
        print("error: {} at epoch : {}".format(err / len(train), i))
    return net
    """error = 0
    for i in range(len(test)):
        net.setInput(test[i])
        net.feedForward()
        net.backPropagate(desired_out_test[i])
        res = net.getResults()
        if (res[0] > 0.5 and desired_out_test[i][0] == 1) or \
                (res[0] < 0.5 and desired_out_test[i][0] == 0):
                error += 1
        print("Network predicted \033[1m{:.3f}\033[0m for \033[1m{}\033[0m".format(res[0], desired_out_test[i][0]))
    print("Correctly predicted {} out of {}".format(error, i))
    print("Percision = {}%".format((error / len(test)) * 100))"""

class Connection:
    """Will keep the infromation between the connection of two neurons"""
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = [] #list of connections
        self.error = 0.0
        self.gradient = 0.0 #value of step we take during gradient descent
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        """update network error"""
        self.error = self.error + err

    def sigmoid(self, x):
        """activation function"""
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        """sigmoid activation derivate for backpropagation"""
        return x * (1.0 - x)

#-------------------------------------------#
#           Setteurs & Getteurs             #

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output
#                                           #
#-------------------------------------------#

    def feedForward(self):
        sumOutput = 0
        #Check if there is any previously connected neuron
        #if not it means its the input or a bias neuron therefor
        #we don't need to feedforward
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            #getting the output of the connected neurons one by one and doing dot product
            #with the weights
            sumOutput += dendron.connectedNeuron.output * dendron.weight
            #sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        #using error * sigmoid_derivate to get gradient step
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            #loop through all the dendrons and calculate and apply all the changes in weights to
            #the current weight
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient);
        self.error = 0;


class Network:
    def __init__(self, topology):
        #topology is network layers template ex : [30, 20, 20, 1]
        #-> 30 neuron in input, 20 and 20 in hidden layers and 1 for output layer
        self.layers = []
        #fill layers in regard to the network topology
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                #first layer, there is no previous layer so we init the neurons with None
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].output = 1
            #layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].output = inputs[i]
            #self.layers[0][i].setOutput(inputs[i])

    def feedForward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForward();

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].error = target[i] - self.layers[-1][i].output
            #self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            #e = (target[i] - self.layers[-1][i].getOutput())
            #err = err + e ** 2
            e = target[i] - self.layers[-1][i].output
            err += e**2
        #err = err / len(target)
        err /= len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            #output.append(neuron.getOutput())
            output.append(neuron.output)
        output.pop()
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.output
            #o = neuron.getOutput()
            if o > 0.5:
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output
