import numpy as np
import cv2
import re
import numpy as np




def main():
    rp = np.loadtxt("transform_robot.txt")
    cp = np.loadtxt("transform_pointscam.txt")

    _,trans, p = cv2.estimateAffine3D(cp.T, rp.T, confidence=0.90, ransacThreshold=200)
    print("p: \n", p)

    cp_new = np.append(cp, np.ones(8))
    cp_new = np.resize(cp_new, (4, 8))

    rp2 = trans@cp_new
    err = rp2 - rp
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/8)
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()


