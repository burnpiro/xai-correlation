import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def plot_bargraph_with_groupings(df, groupby, title, xlabel, ylabel, figsize=(10,20)):
    """
    Plots a dataframe showing the frequency of datapoints grouped by one column and coloured by another.
    df : dataframe
    groupby: the column to groupby
    title: the graph title
    xlabel: the x label,
    ylabel: the y label
    """

    # Makes the bargraph.
    ax = (
        df[groupby]
        .value_counts()
        .plot(kind="barh", figsize=figsize, title=title)
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_images(df, base_path, rows=2, columns=3):
    w = 10
    h = 10
    fig = plt.figure(figsize=(columns*5, rows*5+1))

    # prep (x,y) for extra plotting
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # ax enables access to manipulate each of subplots
    ax = []
    images_df = df.sample(n=columns*rows).reset_index(drop=True)
    for index, row in images_df.iterrows():
        img = mpimg.imread(os.path.join(base_path, row['image']))
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, index+1) )
        ax[-1].set_title("Label:"+str(row['label']))  # set title
        plt.imshow(img, alpha=0.9)

    plt.show()