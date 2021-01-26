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
    names = ['ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', \
            'Concavity', 'Concave points', 'Symetry', 'Fractal Dimension']
    lst_names = list()
    for i in range(2, 32):
        if i >= 2 and i < 13:
            lst_names.append(names[i - 10] + " Mean")
        elif i >= 13 and i < 23:
            lst_names.append(names[i - 20] + " SE (standard error)")
        else:
            lst_names.append(names[i - 30] + " Worst")
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
