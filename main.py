import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import sys
from copy import deepcopy
from lib import utils, neural_network as nn

if __name__ == "__main__":
    args = utils.arg_parse()
    reader = utils.get_csv_infos()
    if args.graph:
        tmp_reader = deepcopy(reader)
        c = 'black'
        for k in reader:
            if "ID" in k:
                pass
            else:
                if args.color:
                    c = utils.parse_color_hist(args.color)
                tmp_reader[str(k)].hist(color=c, alpha=0.5)
                plt.title(str(k))
                plt.xlabel("Value")
                plt.ylabel("Number of elements")
                plt.show()
    del reader['ID'] ##

    values = list(range(len(reader['Diagnosis'])))
    desired_outputs = []
    opposite = 0
    for i in range(len(reader['Diagnosis'])):
        values[i] = 0 if reader['Diagnosis'][i] == 'B' else 1
        #opposite = 1 if values[i] == 0 else 0
        desired_outputs.append([values[i]])
    reader['Diagnosis'] = values
    del reader['Diagnosis']

    train, test = utils.separate_data(reader)

    desired_train = desired_outputs[0:400]
    desired_test = desired_outputs[400::]

    network = None
    if not args.train and not args.predict:
        print("You need to either specify train or predict mode")
        sys.exit(0)
    if args.train:
        network = nn.main_neural(train, desired_train,70)
    elif args.predict and network is not None:
        print("bonsoir")
        predict.main_predict(desired_test, test, network)
