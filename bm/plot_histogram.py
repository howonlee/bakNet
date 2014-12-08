import matplotlib.pyplot as plt
import sys
import collections
import operator

if __name__ == "__main__":
    hist1 = collections.Counter()
    hist2 = collections.Counter()
    with open("sa_res", "r") as f:
        for line in f:
            val = int(line)
            hist1[val] += 1
    with open("eo_res", "r") as f:
        for line in f:
            val = int(line)
            hist2[val] += 1
    cts1 = hist1.most_common()
    cts2 = hist2.most_common()
    print cts1
    print "================"
    print cts2
    plt.loglog(map(operator.itemgetter(0), cts1), map(operator.itemgetter(1), cts1), "r.")
    plt.loglog(map(operator.itemgetter(0), cts2), map(operator.itemgetter(1), cts2), "b.")
    plt.xlabel("number of iterations in trial")
    plt.ylabel("number of trials")
    plt.show()
