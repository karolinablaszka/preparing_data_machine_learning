import numpy as np
from numpy import array
import pandas as pd
import os
from matplotlib import pyplot as plt

def main():
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    list_y = [y1, y2, y3, y4]

    arr_list_y = array(list_y)

    mean_y = np.mean(arr_list_y, axis=1).round(2)
    var_y = np.var(arr_list_y, axis=1).round(2)
    std_y = np.std(arr_list_y, axis=1).round(2)
    pearson_y = np.corrcoef(list_y).round(2)

    results_y = pd.DataFrame([mean_y, var_y, std_y, pearson_y], columns=['y1', 'y2', 'y3', 'y4'])

    path="C:\\Users\\karol\\machine_learning\\data_results\\"
    results_y.to_csv(os.path.join(path,r'results.csv'), sep=',',)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].scatter(x, y1)
    axs[1, 0].scatter(x, y2)
    axs[0, 1].scatter(x, y3)
    axs[1, 1].scatter(x4, y4)
    plt.show()
    fig.savefig(path+'plots.jpg')

if __name__=='__main__':
    main()