import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

def main():

    depth = np.loadtxt("./depth/3.txt")[150:380, 200:800]

    plt.matshow(depth, cmap=plt.cm.viridis)
    plt.clim(0.7, 1)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
