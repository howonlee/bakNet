import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    mat = np.zeros((10,10))
    with open("res", "r") as resfile:
        lines = resfile.readlines()
        for line in lines:
            corr, given = map(int,line.split())
            mat[corr,given] += 1
    plt.imshow(mat, interpolation="none", cmap="gray")
    plt.colorbar()
    plt.xlabel("given")
    plt.ylabel("corr")
    plt.show()
