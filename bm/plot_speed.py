import matplotlib.pyplot as plt
import sys
import collections
import operator

if __name__ == "__main__":
    hist1 = collections.Counter()
    hist2 = collections.Counter()
    with open("sa_time", "r") as f:
        prev = int(f.readline())
        for line in f:
            val = int(line)
            hist1[abs((val - prev) // 10000)] += 1
    with open("eo_time", "r") as f:
        prev = int(f.readline())
        for line in f:
            val = int(line)
            hist2[abs((val - prev) // 10000)] += 1
    cts1 = hist1.most_common()
    cts2 = hist2.most_common()
    plt.plot(map(operator.itemgetter(0), cts1), map(operator.itemgetter(1), cts1), "r.")
    plt.xlabel("number of microseconds per trial")
    plt.ylabel("number of trials")
    plt.show()
    plt.clf()
    plt.plot(map(operator.itemgetter(0), cts2), map(operator.itemgetter(1), cts2), "b.")
    plt.xlabel("number of microseconds per trial")
    plt.ylabel("number of trials")
    plt.show()
