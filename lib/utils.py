import pandas as pd
import argparse
import sys

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

def separate_data(data):
    train = data.iloc[0:400]
    true_train = []
    for i in range(len(train)):
        vec = []
        for j in range(len(train.iloc[i])):
            vec.append(train.iloc[i][j])
        true_train.append(vec)
    test = data.iloc[400:]
    true_test = []
    for i in range(len(test)):
        vec = []
        for j in range(len(test.iloc[i])):
            vec.append(test.iloc[i][j])
        true_test.append(vec)
    return true_train, true_test

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', action='store_true', help='Print historgrams')
    parser.add_argument('-c', '--color', action='store', help='histogram color')
    parser.add_argument('-p', '--plot', action='store_true', help='print plot')
    options = parser.parse_args()
    return options

def append_new_array(base_array, new_array):
    new_array_casted = []
    for i in range(len(new_array)):
        new_array_casted.append(float(new_array[i]))
    base_array.append(new_array_casted)
    return base_array

def create_multi_dim_array(nb_i_dim, size_j_dim, value):
    new_array = []
    for i in range(nb_i_dim):
        new_array.append([])
        for j in range(size_j_dim):
            new_array[i].append(value)
    return new_array

def create_empty_array(nb_elems, value):
    x = value
    array = [x for i in range(nb_elems)]
    return array

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
