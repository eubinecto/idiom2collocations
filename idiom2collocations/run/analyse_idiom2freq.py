import numpy as np

from idiom2collocations.loaders import load_idiom2freq
from matplotlib import pyplot as plt


def main():
    # get the mean frequency.
    idiom2freq = load_idiom2freq()
    freqs = [freq for _, freq in idiom2freq]
    print("mean freq:", str(sum(freqs) / len(freqs)))

    y = np.array(freqs)
    x = np.array(range(len(freqs)))
    plt.title("The frequency of idioms in COCA (spoken) and Opensubtitiles")
    plt.ylabel("frequency of idioms")
    plt.xlabel("idioms by rank order")
    plt.scatter(x, y, s=4)
    plt.show()


if __name__ == '__main__':
    main()
