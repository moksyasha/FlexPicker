import numpy as np
import cv2
import re
import numpy as np


def main():
    transform = np.loadtxt("transform.txt")
    print("Got transform: ", transform)
    camera = np.array([301.,         227.,           1.15100002])
    camera = np.append(camera, [1])
    #print(np.linalg.det(transform[:, :3]))
    #transform = np.delete(transform, (3), axis=0)
    #transform = np.append(transform, np.array([0, 0, 0, 1]))
    #transform.resize(4, 4)
    print(transform)
    result = np.dot(transform, camera)
    result = np.delete(result, -1)
    print(result)


if __name__ == "__main__":
    main()


