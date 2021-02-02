import pandas as pd
import argparse
import sys

def arg_parse():
    parser = argparse.ArgumentParser()
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
