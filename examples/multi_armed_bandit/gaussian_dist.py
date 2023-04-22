from matplotlib import pyplot as plt
import numpy as np


def gaussian_dist():
    statistic_data = np.random.randn(2000, 10) # [2000, 10]
    plt.violinplot(dataset=statistic_data + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('images/gaussian_dist.png')
    plt.close()

gaussian_dist()