import numpy as np
import matplotlib.pyplot as plt
import sys, re, math

if __name__ == "__main__":
    non_decimal = re.compile(r'[^-\d.]+')
    with open("eo_spin", "r") as f:
        errs = map(lambda x: non_decimal.sub("", x), f.read().split())
        errs = map(float, errs)
        cum_diffs = []
        for idx, err in enumerate(errs):
            if idx > 1:
                cum_diffs.append(abs(errs[idx] - errs[idx-1]))
        plt.semilogy(cum_diffs, "r.")
    with open("tau_eo_spin", "r") as f:
        errs = map(lambda x: non_decimal.sub("", x), f.read().split())
        errs = map(float, errs)
        cum_diffs = []
        for idx, err in enumerate(errs):
            if idx > 1:
                cum_diffs.append(abs(errs[idx] - errs[idx-1]))
        plt.semilogy(cum_diffs)
    plt.show()
