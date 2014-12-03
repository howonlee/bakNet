import numpy as np
import matplotlib.pyplot as plt
import sys, re, math

if __name__ == "__main__":
    non_decimal = re.compile(r'[^-\d.]+')
    with open("eo_all", "r") as f:
        errs = map(lambda x: non_decimal.sub("", x), f.read().split())
        errs = map(float, errs)
        plt.plot(errs,"r-", label="EO")
    with open("tau_eo_all", "r") as f:
        errs = map(lambda x: non_decimal.sub("", x), f.read().split())
        errs = map(float, errs)
        plt.plot(errs, label="tEO")
    plt.title("Progression of Ising model energy")
    plt.xlabel("epoch")
    plt.ylabel("energy")
    plt.legend()
    plt.show()
