###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
import numpy as np


def plot_side_by_side_bars(bar_1, bar_2, label_1, label_2, x_label, y_label, title, x_ticks):
    # data to plot
    n_groups = len(bar_1)
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, bar_1, bar_width,
                     alpha=opacity,
                     color='b',
                     label=label_1)

    rects2 = plt.bar(index + bar_width, bar_2, bar_width,
                     alpha=opacity,
                     color='g',
                     label=label_2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(index + bar_width, x_ticks)
    plt.legend()
    plt.tight_layout()
    plt.show()