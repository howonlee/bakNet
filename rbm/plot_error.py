import numpy as np
import matplotlib.pyplot as plt
import sys, re

if __name__ == "__main__":
    non_decimal = re.compile(r'[^\d.]+')
    with open("taueo_error", "r") as f:
        errs = map(lambda x: non_decimal.sub("", x), f.read().split())
        errs = map(float, errs)
        plt.plot(errs, "r.")
    with open("opt_tau_eo_error", "r") as f:
        errs = map(lambda x: non_decimal.sub("", x), f.read().split())
        errs = map(float, errs)
        plt.plot(errs)
    plt.show()
