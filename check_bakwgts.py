import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.sort(np.load("bak_out_wgts.npy").ravel())
    plt.loglog(range(len(data)), data)
    plt.show()
