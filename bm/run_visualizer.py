import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    assert len(sys.argv) > 1
    ls = []
    with open(sys.argv[1], "r") as runfile:
        for line in runfile:
            ls.append(map(int, line.split()))
    arr = np.array(ls)
    plt.matshow(arr, cmap=plt.gray())
    plt.ylabel("time")
    plt.xlabel("activation in spot")
    plt.axis([0,16,587,500])
    plt.title("Fast run in SA")
    plt.show()
