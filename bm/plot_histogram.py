import matplotlib.pyplot as plt
import sys
import collections
import operator

if __name__ == "__main__":
    assert len(sys.argv) > 1
    hist = collections.Counter()
    with open(sys.argv[1], "r") as f:
        for line in f:
            val = int(line)
            hist[val] += 1
    cts = hist.most_common()
    print cts
    plt.loglog(map(operator.itemgetter(1), cts), map(operator.itemgetter(0), cts), "ro")
    plt.xlabel("number of iterations in trial")
    plt.ylabel("number of trials")
    plt.show()
