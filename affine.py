import numpy as np
import cv2
import re
import numpy as np
def main():
    transform = np.loadtxt("transform.txt")
    print("Got transform: ", transform)
    camera = np.array([ 0.02510429, -0.22548195,  1.03200006])
    camera = np.append(camera, [1])
    print(np.linalg.det(transform[:, :3]))
    transform = np.append(transform, [0, 0, 0, 1])
    transform.resize(4, 4)
    result = np.matmul(transform, camera)
    result = np.delete(result, -1)
    print(result)


if __name__ == "__main__":
    main()


