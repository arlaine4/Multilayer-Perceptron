import pandas as pd
import numpy as np
from sklearn import preprocessing
import argparse
import sys

def export_network_data(layers):
    lst_elems = []
    for i in range(len(layers)):
        print("Layers[{}] : ".format(i))
        for elem in layers[i]:
            print("Output : ", elem.output)
            for dendron in elem.dendrons:
                print("weight : {}\ndWeight : {}\nconnectedNeuron : \n\t{}\n\t{}\n\t{}".format(dendron.weight, dendron.dWeight, dendron.connectedNeuron.output, dendron.connectedNeuron.error, dendron.connectedNeuron.gradient))
    return lst_elems

def get_min_max_scale(data):
    min_ = 0
    max_ = 0
    for i in range(len(data)):
        for j in range(len(data.iloc[i])):
            if data.iloc[i][j] > max_:
                max_ = data.iloc[i][j]
            elif data.iloc[i][j] < min_:
                min_ = data.iloc[i][j]
    return [min_, max_]

def scale_input_data(data):
    input_a = np.array(data)
    input_a = input_a.reshape(-1, 1)
    mean_all = preprocessing.PowerTransformer()
    mean_all = mean_all.fit(input_a)
    mean_all = mean_all.transform(input_a)
    mean_all = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    new_mean_all = mean_all.fit_transform(input_a)
    last_values = []
    for i in range(len(new_mean_all)):
        last_values.append(float(new_mean_all[i]))
    return last_values


def separate_data(data):
    train = data.iloc[0:400]
    true_train = []
    for i in range(len(train)):
        vec = []
        for j in range(len(train.iloc[i])):
            vec.append(train.iloc[i][j])
        vec = scale_input_data(vec)
        true_train.append(vec)
    test = data.iloc[400:]
    true_test = []
    for i in range(len(test)):
        vec = []
        for j in range(len(test.iloc[i])):
            vec.append(test.iloc[i][j])
        vec = scale_input_data(vec)
        true_test.append(vec)
    return true_train, true_test

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train the network')
    parser.add_argument('-p', '--predict', action='store_true', help='launch network prediction test')
    parser.add_argument('-g', '--graph', action='store_true', help='Print historgrams')
    parser.add_argument('-c', '--color', action='store', help='histogram color')
    options = parser.parse_args()
    return options

def get_csv_infos():
    lst_names = ['ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', \
            'Concavity', 'Concave points', 'Symetry', 'Fractal Dimension', 'Radius Mean', \
            'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', \
            'Concave Points Mean', 'Symetry Mean', 'Fractal Dimension Mean', 'Radius Worst', \
            'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', \
            'Concave Points Worst', 'Symetry Worst', 'Fractal Dimension Worst']
    reader = pd.read_csv("data.csv", names = lst_names)
    try:
        reader = pd.read_csv("data.csv", names=lst_names)
    except:
        print("Error loading dataset.")
        sys.exit(0)
    return reader

def parse_color_hist(color):
    if color == 'r':
        return 'red'
    elif color == 'b':
        return 'blue'
    elif color == 'y':
        return 'yellow'
    elif color == 'g':
        return 'green'
    elif color == 'o':
        return 'orange'
    elif color == 'c':
        return 'cyan'
    elif color == 'm':
        return 'magenta'
    else:
        return 'black'
