import matplotlib.pyplot as plt
import sys
import collections
import operator

if __name__ == "__main__":
    hist1 = collections.Counter()
    hist2 = collections.Counter()
    hist3 = collections.Counter()
    with open("sa_res", "r") as f:
        for line in f:
            val = int(line)
            hist1[val] += 1
    with open("eo_res", "r") as f:
        for line in f:
            val = int(line)
            hist2[val] += 1
    with open("sched_res", "r") as f:
        for line in f:
            val = int(line)
            hist3[val] += 1
    cts1 = hist1.most_common()
    cts2 = hist2.most_common()
    cts3 = hist3.most_common()
    plt.loglog(map(operator.itemgetter(0), cts1), map(operator.itemgetter(1), cts1), "r.", label="Simulated Annealing, T=1", alpha=0.4)
    plt.loglog(map(operator.itemgetter(0), cts2), map(operator.itemgetter(1), cts2), "b.", label="t-Extremal Optimization", alpha=0.4)
    plt.loglog(map(operator.itemgetter(0), cts3), map(operator.itemgetter(1), cts3), "g.", label="Simulated Annealing with Annealing Schedule", alpha=0.4)
    plt.xlabel("number of iterations in trial")
    plt.ylabel("number of trials")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10))
    plt.savefig("iter_hist", bbox_inches="tight")
