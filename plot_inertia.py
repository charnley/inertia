
import matplotlib.pyplot as plt



def main():

    X = [0, 0.5, 1]
    Y = [1, 0.5, 1]

    plt.plot(X, Y)

    plt.plot([0.36],[0.65], 'k.')

    plt.savefig("fig_inertia")

    return

if __name__ == "__main__":
    main()


