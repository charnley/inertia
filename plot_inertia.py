
import copy

import matplotlib.pyplot as plt
import numpy as np
import rmsd
from matplotlib.ticker import NullFormatter
from scipy.stats import gaussian_kde


def save_figure(filename, fig=None):

    if fig is None:

        plt.savefig(filename + ".png", bbox_inches="tight")
        plt.savefig(filename + ".pdf", bbox_inches="tight")

    else:

        fig.savefig(filename + ".png", bbox_inches="tight")
        fig.savefig(filename + ".pdf", bbox_inches="tight")

    return


def get_ratio(inertia):

    inertia.sort()

    ratio = np.zeros(2)
    ratio[0] = inertia[0]/inertia[2]
    ratio[1] = inertia[1]/inertia[2]

    return ratio


def get_gaussian_kernel(xvalues):

    bins = np.linspace(0.0,1.0, 200)
    gaussian_kernel = gaussian_kde(xvalues)
    values = gaussian_kernel(bins)

    return bins, values


def rotation_matrix(sigma):
    """
    https://en.wikipedia.org/wiki/Rotation_matrix
    """

    radians = sigma * np.pi / 180.0

    r11 = np.cos(radians)
    r12 = -np.sin(radians)
    r21 = np.sin(radians)
    r22 = np.cos(radians)

    R = np.array([[r11, r12], [r21, r22]])

    return R


def scale_triangle_with_kde(xvalues, yvalues, filename="_istwk"):


    fig_kde, axes_kde = plt.subplots(3, sharex=True, sharey=True)
    fig_his, ax_his = plt.subplots(1)

    # define edges
    sphere = np.array([1, 1])
    rod = np.array([0, 1])
    disc = np.array([0.5, scale_func(0.5)])
    sphere = sphere[np.newaxis]
    rod = rod[np.newaxis]
    disc = disc[np.newaxis]

    # define and scale coord for distances to sphere
    yvalues_scale = scale_func(copy.deepcopy(yvalues))
    coord = np.array([xvalues, yvalues_scale])

    linewidth=0.9

    dots = [sphere, rod, disc]
    names = ["Sphere", "Rod", "Disc"]

    for ax, dot, name in zip(axes_kde, dots, names):

        dist = distance(coord, dot.T)
        bins, values = get_gaussian_kernel(dist)
        ax.plot(bins, values, "k", linewidth=linewidth, label=name)

        ax.text(0.045,0.75,name,
            horizontalalignment='left',
            transform=ax.transAxes)

        ax.get_yaxis().set_visible(False)

    ax = axes_kde[-1]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["is shape", "not shape"])

    # prettify
    fig_kde.subplots_adjust(hspace=0)

    # save
    save_figure(filename+"_kde", fig=fig_kde)

    # His
    nbins = 100
    H, xedges, yedges = np.histogram2d(xvalues, yvalues, bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)

    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero

    pc = ax_his.pcolormesh(xedges,yedges,Hmasked, cmap="PuRd")
    ax_his.set_aspect('equal')

    ax_his.get_yaxis().set_visible(False)
    ax_his.get_xaxis().set_visible(False)
    ax_his.spines['top'].set_visible(False)
    ax_his.spines['right'].set_visible(False)
    ax_his.spines['bottom'].set_visible(False)
    ax_his.spines['left'].set_visible(False)

    ax_his.set_ylim([0.5-0.05, 1.05])
    ax_his.set_xlim([0.0-0.05, 1.0+0.05])

    max_count = np.max(H)
    max_count /= 10
    max_count = np.floor(max_count)
    max_count *= 10

    cb_ticks = np.linspace(1, max_count, 3, dtype=int)
    # cb_ticks = [1, max_count]

    cb = fig_his.colorbar(pc, orientation="horizontal", ax=ax_his, ticks=cb_ticks, pad=0.05)
    cb.outline.set_edgecolor('white')


    # prettify
    ax_his.text(1.0, 1.02, "Sphere (1,1)",
        horizontalalignment='center')
    ax_his.text(0.0, 1.02, "Rod (0,1)",
        horizontalalignment='center')
    ax_his.text(0.5, 0.5- 0.04, "Disc (0.5,0.5)",
        horizontalalignment='center')

    save_figure(filename + "_his", fig=fig_his)

    return


def scale_func(Y):
    """

    scale
    1.0 - 0.5
    too
    1.0 - 0.8660254037844386

    """

    target_y = 0.133975
    factor_y = 1.0 + ( -target_y)/0.5

    diff = Y - 1.0
    add = diff*factor_y
    Y += add

    return Y


def distance(coord, dot):
    dist = coord - dot
    dist = dist**2
    dist = np.sum(dist, axis=0)
    dist = np.sqrt(dist)

    return dist


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="csv")
    args = parser.parse_args()

    # get name
    name = args.filename.split(".")
    name = ".".join(name[:-1])

    # Read csvfile
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
        ratio = get_ratio(line)

        if sum(line) == 0:
            print("zero sum", i+1)
            continue

        X.append(ratio[0])
        Y.append(ratio[1])

    X = np.array(X)
    Y = np.array(Y)


    # what to dooo
    scale_triangle_with_kde(X, Y, filename=name)




    return


def test():

    name = args.filename.split(".")
    name = ".".join(name[:-1])
    print(name)

    # Triangle
    X = [0, 0.5, 1, 0]
    Y = [1, 0.5, 1, 1]
    tri = [X, Y]
    tri = np.array(tri)
    # plt.plot(X, Y, linewidth=0.5, color="grey")

    X = [0, 0.5, 1]
    Y = [1.0, 0.5, 1.0]
    R = np.array([X, Y]).T


    cent_tri = rmsd.centroid(R)
    print(cent_tri)

    X = np.array(X)
    Y = np.array(Y)



    Y = scale_func(Y)

    print("scale")
    print(Y)

    coord = np.array([X, Y]).T
    plt.plot(X, Y, "rx")

    cent_tri = rmsd.centroid(coord)
    print("center:", cent_tri)

    plt.plot(*cent_tri, "ko")
    plt.plot(X, Y)

    sphere = np.array([1, 1])
    rod = np.array([0, 1])
    disc = np.array([0.5, scale_func(0.5)])

    plt.plot(*sphere, "o")


    dist = distance(cent_tri, sphere)
    dist = distance(cent_tri, rod)
    dist = distance(cent_tri, disc)

    # plt.ylim([0.0, 1.0])
    # plt.xlim([0.0, 1.0])
    # plt.savefig("test")


    # tri = tri.T
    # tri -= cent_tri
    # tri = np.dot(tri, rotation_matrix(45.0))
    # tri += cent_tri

    # print(max(tri[0]))
    # print(min(tri[0]))
    #
    # plt.plot(*tri.T, "g-")


    # rotate example
    # dot_to_rot = [[1.0], [1.0]]
    # dot_to_rot = np.array(dot_to_rot)
    #
    # plt.plot(*dot_to_rot, "r.")
    # dot_to_rot = dot_to_rot.T
    # dot_to_rot -= cent_tri
    #
    # dot_to_rot = np.dot(dot_to_rot, rotation_matrix(30))
    # dot_to_rot += cent_tri
    #
    # plt.plot(*dot_to_rot.T, "g.")



    # sphere = np.array([[1],[1]])
    # dist = R - sphere
    # dist = dist**2
    # dist = np.sum(dist, axis=0)
    # dist = np.sqrt(dist)
    # print(dist)



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
        ratio = get_ratio(line)

        # if np.abs(ratio[0] - 0) < 0.01 and \
        #     np.abs(ratio[1] - 1) < 0.01:
        #         print(smi)

        if sum(line) == 0:
            print("zero sum", i+1)
            continue

        X.append(ratio[0])
        Y.append(ratio[1])


    X = np.array(X)
    Y = np.array(Y)


    Y = scale_func(Y)

    coord = np.array([X, Y])

    sphere = sphere[np.newaxis]
    rod = rod[np.newaxis]
    disc = disc[np.newaxis]


    plt.clf()
    dist = distance(coord, sphere.T)
    bins, values = get_gaussian_kernel(dist)
    plt.plot(bins, values, linewidth=1.0, label="sphere")

    dist = distance(coord, rod.T)
    bins, values = get_gaussian_kernel(dist)
    plt.plot(bins, values, linewidth=1.0, label="rod")

    dist = distance(coord, disc.T)
    bins, values = get_gaussian_kernel(dist)
    plt.plot(bins, values, linewidth=1.0, label="disc")

    plt.legend(loc="best")

    plt.savefig("tmp_fig_kde_dist_"+name + ".png")
    plt.clf()



    # hb = plt.hexbin(X, Y, gridsize=70, cmap='gist_heat_r', zorder=2)
    # cb = plt.colorbar(hb)
    # cb.set_label('log10(N)')

    # plt.style.use('seaborn-white')


    # C = np.array([X, Y])
    # C = C.T
    # C -= cent_tri
    # C = np.dot(C, rotation_matrix(-45))
    # C += cent_tri
    # C = C.T
    # X, Y = C


    # View disc-rod histogram
    # idx_view, = np.where(Y < 0.61)
    # print(idx_view)
    # X = X[idx_view]
    # Y = Y[idx_view]



    # fucks up my plot tho
    # plt.hist2d(X, Y, bins=30, cmap='Blues')
    # cb = plt.colorbar()
    # cb.set_label('counts in bin')

    # plt.scatter(X, Y, zorder=2, s=0.01, color="k")


    # Estimate the 2D histogram
    nbins = 100
    H, xedges, yedges = np.histogram2d(X, Y, bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    plt.pcolormesh(xedges,yedges,Hmasked, cmap="Blues")
    cb = plt.colorbar()
    cb.set_label('counts in bin')


    # plt.ylim([-1, 1.5])
    # plt.xlim([-1, 1.5])
    #
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])



    # plt.axis('off')

    plt.savefig("tmp_fig_inhis_"+name + ".png", bbox_inches='tight')
    plt.clf()

    from scipy.stats import gaussian_kde

    # bins = np.linspace(0.02, 0.74, 200)
    bins = np.linspace(0.0,1.0, 200)
    gaussian_kernel = gaussian_kde(X)
    values = gaussian_kernel(bins)
    plt.plot(bins, values, "k", linewidth=1.0)
    plt.savefig("tmp_fig_kde_"+name + ".png")


    return


if __name__ == "__main__":
    main()
