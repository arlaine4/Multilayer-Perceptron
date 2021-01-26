import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
from copy import deepcopy
import utils

if __name__ == "__main__":
    args = utils.arg_parse()
    reader = utils.get_csv_infos()
    if args.graph:
        tmp_reader = deepcopy(reader)
        c = 'black'
        for k in reader:
            if "ID" not in k: #Pass some useless histographs to show like ID
                if args.color:
                    c = utils.parse_color_hist(args.color)
                tmp_reader[str(k)].hist(color=c, alpha=0.5)
                plt.title(str(k))
                plt.xlabel("Value")
                plt.ylabel("Number of elements")
                plt.show()
            else:
                pass
    else:
        print(reader)
