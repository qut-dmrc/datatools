""" An opinionated set of functions to help generate nice looking plots
"""
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 10)


def init_plotting_defaults(working_dir=None):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)

    # Set STDERR handler as the only handler
    logger.handlers = [handler]

    # Don't show scientific notation for pandas
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    if working_dir:
        os.chdir(working_dir)

    sns.set_context("paper")
    sns.set_style("ticks")
    plt.style.use(['seaborn-deep', 'seaborn-paper'])
    sns.despine()

    sns.set_context("paper", rc={})

    params = {
        'legend.fontsize': 8,
        'text.usetex': False,
        'figure.figsize': [8, 4],
        'font.family': "serif",
        'font.size': 10,
        "axes.titlesize": 8,
        "axes.labelsize": 6,
        'figure.dpi': 2 * 163,  # 163 = 4k 27", #109 = ?; 227 = MBP screen,
        'savefig.dpi': 300,
        'savefig.directory': working_dir,
        "font.serif": "Roboto Slab",
        'figure.titlesize': 16,
        'lines.linewidth': 6,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        "font.sans-serif": "Roboto",
        "font.family": "Roboto",
        "font.weight": 'light',
#        "figure.figsize": (6, 2),
        'axes.titleweight': 'medium',

    }

    plt.rcParams.update(params)


def legend_bottom(ax):
    # Shrink current axis's height by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * .2,
                     box.width, box.height *.8])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=False, shadow=False, ncol=2)
    return l


def autolabel(ax, barh=False, decimals=2, suffix="", convert_to_percent=True, font_size=None):
    """ Automatically add labels to bar charts in matplotlib
        Code initially from http://composition.al/blog/2015/11/29/a-better-way-to-add-labels-to-bar-charts-with-matplotlib/
    """
    if not font_size:
        font_size = plt.rcParams['font.size']

    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom
    (x_bottom, x_top) = ax.get_xlim()
    x_width = x_top - x_bottom

    if barh:  # horizontal plot
        for p in ax.patches:
            width = p.get_width()

            # Fraction of axis height taken up by this rectangle
            p_width = (width / x_width)

            if convert_to_percent:
                ax_label = np.round(p.get_width() * 100, decimals=decimals)
            else:
                ax_label = np.round(p.get_width(), decimals=decimals)

            # If we can fit the label above the column, do that;
            # otherwise, put it inside the column.
            if p_width > 0.7:  # arbitrary
                label_position = width - (x_width * 0.3)
            else:
                label_position = width + (x_width * 0.01)

            ax.annotate("{}{}".format(ax_label, suffix),
                        (label_position, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=font_size)
    else:
        for p in ax.patches:
            height = p.get_height()

            # Fraction of axis height taken up by this rectangle
            p_height = (height / y_height)

            if convert_to_percent:
                ax_label = np.round(p.get_height() * 100, decimals=decimals)
            else:
                ax_label = np.round(p.get_height(), decimals=decimals)

            # If we can fit the label above the column, do that;
            # otherwise, put it inside the column.
            if p_height > 0.7:
                label_position = height - (y_height * 0.3)
            else:
                label_position = height + (y_height * 0.01)

            ax.annotate("{}{}".format(ax_label, suffix),
                        (p.get_x() + p.get_width() / 2., label_position),
                        ha='center', va='bottom', fontsize=font_size)


def reduce_xticks(list_of_axes, n=2):
    # iterate over axes of FacetGrid
    for ax in list_of_axes:
        labels = ax.get_xticklabels()  # get x labels
        for i, l in enumerate(labels):
            if (i % n != 0): labels[i] = ''
        ax.set_xticklabels(labels, rotation=30)  # set new labels

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    From https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
