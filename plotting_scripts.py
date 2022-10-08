import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import classes as cl
import networkx as nx
import functions as fn


def plotxy(data_list, telescope):
    """Plot positions of a telescope and its SVD errors for a list of data"""
    names = [data.name for data in data_list]
    xvals = []
    yvals = []
    xerr = []
    yerr = []

    for data in data_list:
        if telescope in data.telescopes.unique_telescopes:
            position = data.svd.x('telescopes', telescope=telescope)
            error = data.svd.std('telescopes', telescope=telescope)
            xvals.append(position[0])
            yvals.append(position[1])
            xerr.append(error[0])
            yerr.append(error[1])
        else:
            names.remove(data.name)
    
    # Plot around mean
    xvals = np.array(xvals)
    xvals -= xvals.mean()
    yvals = np.array(yvals)
    yvals -= yvals.mean()

    plt.figure(figsize=(10,10))
    plt.errorbar(xvals, yvals, xerr=xerr, yerr=yerr, fmt='o', ms=1, capsize=2)
    for i, txt in enumerate(names):
        plt.annotate(txt, (xvals[i], yvals[i]))
    
    plt.title(f"x, y coordinates of telescope {telescope} around average")
    plt.xlabel("x/m")
    plt.ylabel("y/m")


def plotz(data_list, telescope):
    names = [data.name for data in data_list]
    zvals = []
    zerr = []

    for data in data_list:
        if telescope in data.telescopes.unique_telescopes:
            position = data.svd.x('telescopes', telescope=telescope)
            error = data.svd.std('telescopes', telescope=telescope)
            zvals.append(position[2])
            zerr.append(error[2])
        else:
            names.remove(data.name)
    
    # Plot around mean
    zindex = np.arange(len(names))
    zvals = np.array(zvals)
    zvals -= zvals.mean()

    plt.figure(figsize=(10,5))
    plt.errorbar(zindex, zvals, yerr=zerr, fmt='o', ms=1, capsize=2, color='black')
    plt.bar(zindex, zvals)
    plt.xticks(ticks=zindex-0.5, labels=names, rotation=45)
    
    plt.title(f"z coordinate of telescope {telescope} around average")
    plt.ylabel("z/m")


# Code to show 1 decimal place in x and y-axis ticks while using scientific notation
from matplotlib.ticker import ScalarFormatter
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

def plot_positions(pair_list, telescope, ref_data):
    telescopes = ref_data.telescopes.unique_telescopes[1:]
    i = telescopes.index(telescope)
    means_x = []
    means_y = []
    means_z = []
    for pair in pair_list:
        x, y, z = fn.month_mean(pair[0], pair[1])[0][i]
        means_x.append(x)
        means_y.append(y)
        means_z.append(z)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8), gridspec_kw={'width_ratios': [2.5, 1]})

    pairnames = [f"{pair[0].name}/{pair[1].name}" for pair in pair_list]
    for pairnumber, pair in enumerate(pair_list):
        mean, err = fn.month_mean(pair[0], pair[1])
        x, y, z = mean[i]
        x -= np.array(means_x).mean()
        y -= np.array(means_y).mean()
        z -= np.array(means_z).mean()
        sx, sy, sz = err[i]
        pairname = pairnames[pairnumber]
        ax1.errorbar(x, y, xerr=sx, yerr=sy, label=pairname, lw=5)
        ax2.errorbar(pairnumber, z, yerr=sz, label=pairname, fmt='o', ms=3, capsize=3, lw=5)

    # Formatting
    ax1.set_xlabel("x/m")
    ax1.set_ylabel("y/m")
    ax2.set_ylabel("z/m")
    ax2.set_xticks([])
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlim(-0.5, len(pair_list)-0.5)
    ax1.set_title(f"{telescope}")
    #ax1.legend()
    fmt = ScalarFormatterForceFormat()
    fmt.set_powerlimits((0,0))
    ax1.xaxis.set_major_formatter(fmt)
    ax1.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_formatter(fmt)
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.rcParams.update({'font.size': 30, 'axes.linewidth':2, 'xtick.major.width':3, 'ytick.major.width':3, 'xtick.major.size':8, 'ytick.major.size':8})
    ax1.legend(prop={'size': 15})

