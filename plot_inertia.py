
import matplotlib.pyplot as plt

import calculate_inertia as cain


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help="Calculate inertia of filename.{.sdf.gz,.smi.gz,.sdf,smi}")
    parser.add_argument('-j', '--procs', type=int, help="Use subprocess to run over more cores", default=1)
    args = parser.parse_args()


    X = [0, 0.5, 1]
    Y = [1, 0.5, 1]

    plt.plot(X, Y)

    plt.plot([0.36],[0.65], 'k.')

    plt.savefig("fig_inertia")

    return

if __name__ == "__main__":
    main()


