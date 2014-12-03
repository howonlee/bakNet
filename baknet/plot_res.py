import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("bak_res_3", "r") as f:
        errs = f.read().split()
        trains = []
        tests = []
        for idx, x in enumerate(errs):
            if idx % 2 == 0:
                trains.append(float(x))
            else:
                tests.append(float(x))
        plt.plot(trains, "r-", label="training")
        plt.plot(tests, "b-", label="testing")
        plt.ylim([0,1])
        plt.title("Errors from Adaptive Extremal Network on MNIST")
        plt.ylabel("error")
        plt.xlabel("epoch / 200")
        plt.legend(loc=4)
        plt.savefig("bak_plot")
