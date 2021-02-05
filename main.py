import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
from copy import deepcopy
#from lib import utils, maths, neural_network as nn
from lib import utils, maths, neural as nn

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
    for i in range(len(reader['Diagnosis'])):
        values[i] = 0 if reader['Diagnosis'][i] == 'B' else 1
        desired_outputs.append(values[i])
    reader['Diagnosis'] = values
    desired_outputs = reader['Diagnosis']
    del reader['Diagnosis']
    test = nn.NeuralNetwork(reader, desired_outputs, 2, [20, 20], 2, 'train')
    test.fit(reader)
    #test.__str__()
    #test = nn.NeuralNetwork(reader, 2, 10, 2, desired_outputs)
    #test = nn.NeuralNetwork(reader, 2, 10, 2)
    #test.run()
    #test.__str__()
    
    #test_w = [0.3, 1.2, -0.4, 1.1]
    #print("weighted sum for {} = {}".format(test_w, nn.weighted_sum(test_w, 0.75)))
    #----------------------Test math functions-------------------------------#

    """print("Max from numpy.amax() :")
    print(math.max(pd.DataFrame(test).to_numpy()))
    print("Max from lib.math :")
    print(math.max(pd.DataFrame(test).to_numpy(), True), '\n')

    print("Min from numpy.amin() :")
    print(math.min(pd.DataFrame(test).to_numpy()))
    print("Min from lib.math : ")
    print(math.min(pd.DataFrame(test).to_numpy(), True), '\n')

    print("Mean from numpy.mean() : ")
    print(math.mean(pd.DataFrame(test).to_numpy()))
    print("Mean from lib.math : ")
    print(math.mean(pd.DataFrame(test).to_numpy(), True), '\n')

    print("Standard Error from numpy.std() : ")
    print(math.standard_error(pd.DataFrame(test).to_numpy()))
    print("Standard Error from lib.math : ")
    print(math.standard_error(pd.DataFrame(test).to_numpy(), True), '\n')

    print("Standard deviation from numpy.std() / (np.size(array)**0.5) : ")
    print(math.standard_deviation(pd.DataFrame(test).to_numpy()))
    print("Standard deviation from lib.math : ")
    print(math.standard_deviation(pd.DataFrame(test).to_numpy(), True), '\n')

    print("Variance from numpy.std() / numpy.mean() : ")
    print(math.variance(pd.DataFrame(test).to_numpy()))
    print("Variance from lib.math : ")
    print(math.variance(pd.DataFrame(test).to_numpy(), True), '\n')"""
    #------------------------------------------------------------------------#
