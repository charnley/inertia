
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import gaussian_kde

import calculate_inertia as cain


def histo_view(xvalues, yvalues,
        filename="overview_scathis",
        x_binwidth=None,
        y_binwidth=None,
        debug=False):

    from matplotlib import rc

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)

    nullfmt = NullFormatter()

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.1, height]

    plt.figure(1, figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)


    # scatter plot
    # ax_scatter.scatter(xvalues, yvalues, color="k", alpha=0.4)
    hb = ax_scatter.hexbin(xvalues, yvalues, gridsize=40, bins='log', cmap='Greys')

    plt.savefig(filename, bbox_inches="tight")
    plt.savefig(filename + ".pdf", bbox_inches="tight")

    return

    # define binwidth
    x_max = np.max(xvalues)
    x_min = np.min(xvalues)
    x_binwidth = (abs(x_min) + x_max) / 30.0
    x_binwidth = int(x_binwidth)
    x_binwidth = 1
    x_bins = np.arange(x_min, x_max+x_binwidth, x_binwidth)

    y_max = np.max(yvalues)
    y_min = np.min(yvalues)
    y_binwidth = (abs(y_min) + y_max) / 50.0
    y_binwidth = int(y_binwidth)
    y_bins = np.arange(y_min, y_max+y_binwidth, y_binwidth)

    # xlim = (x_min-x_binwidth, x_max+x_binwidth)
    # ylim = (y_min-y_binwidth, y_max+y_binwidth)

    # Set limits and ticks of scatter
    # ax_scatter.set_xlim(xlim)
    # ax_scatter.set_ylim(ylim)

    xkeys = np.arange(10, x_max+x_binwidth*2, 10)
    xkeys = [1] + list(xkeys)
    ykeys = np.arange(0, y_max+y_binwidth, 100)

    # Histogram

    bins = np.linspace(x_min, x_max, 200)
    gaussian_kernel = gaussian_kde(xvalues)
    values = gaussian_kernel(bins)
    ax_histx.plot(bins, values, "k", linewidth=1.0)

    bins = np.linspace(y_min, y_max, 200)
    gaussian_kernel = gaussian_kde(yvalues)
    values = gaussian_kernel(bins)
    ax_histy.plot(values, bins, "k", linewidth=1.0)

    # ax_histx.hist(xvalues, bins=x_bins, histtype='step', color="k")
    # ax_histy.hist(yvalues, bins=y_bins, orientation='horizontal', histtype='step', color="k")

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # pretty
    if not debug:
        ax_histx.xaxis.set_major_formatter(nullfmt)
        ax_histy.yaxis.set_major_formatter(nullfmt)

        set_border(ax_scatter, xkeys, ykeys)
        set_border(ax_histx, [], [], border=[False, False, False, False])
        set_border(ax_histy, [], [], border=[False, False, False, False])

        ax_histx.set_xticks([], [])
        ax_histy.set_yticks([], [])


    # ax_scatter.set_xlabel("Heavy atoms")
    # ax_scatter.set_ylabel("Kelvin")

    plt.savefig(filename, bbox_inches="tight")
    plt.savefig(filename + ".pdf", bbox_inches="tight")

    return



def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="csv")
    args = parser.parse_args()


    # Triangle
    X = [0, 0.5, 1, 0]
    Y = [1, 0.5, 1, 1]
    plt.plot(X, Y, linewidth=0.5)

    f = open(args.filename)

    X = []
    Y = []

    for i, line in enumerate(f):

        line = line.split()

        if len(line) > 3:
            smi = line[0]
            line = line[1:]

        line = [float(x) for x in line]
        line = np.array(line)
        ratio = cain.get_ratio(line)

        # if np.abs(ratio[0] - 0) < 0.01 and \
        #     np.abs(ratio[1] - 1) < 0.01:
        #         print(smi)

        if sum(line) == 0:
            print("zero sum", i+1)
            continue

        X.append(ratio[0])
        Y.append(ratio[1])

    # hb = plt.hexbin(X, Y, gridsize=70, cmap='gist_heat_r', zorder=2)
    # cb = plt.colorbar(hb)
    # cb.set_label('log10(N)')

    plt.scatter(X, Y, zorder=2, s=0.01, color="k")

    name = args.filename.split(".")
    name = ".".join(name[:-1])
    print(name)
    plt.savefig("tmp_fig_inhis_"+name)

    return

if __name__ == "__main__":
    main()


