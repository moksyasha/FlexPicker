import numpy as np
import cv2
import re
import numpy as np


def cam_to_rob(trans, x, y, d):
    cpoint = np.array([x, y, d, 1]).reshape((4, 1))
    return trans@cpoint


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def affine(rp, cp):
    _,trans, p = cv2.estimateAffine3D(cp, rp, confidence = 0.95, ransacThreshold=3)
    print("Points: \n", p)
    cp_new = np.hstack((cp, np.ones((cp.shape[0], 1))))
    rp2 = np.dot(trans, cp_new.T)
    print("Orig: \n", np.array(rp.T).astype(np.int32))

    rp2 = np.dot(trans, cp_new.T)
    print("Palb: \n", np.array(rp2).astype(np.int32))
    err = rp2 - rp.T
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/rp2.shape[0])
    print("RMSE:", rmse)


def rigid(rp, cp):
    print("Orig: \n", np.array(rp).astype(np.int32))
    ret_R, ret_t = rigid_transform_3D(cp, rp)
    rp2 = (ret_R@cp) + ret_t
    print("Palb: \n", np.array(rp2).astype(np.int32))
    err = rp2 - rp
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/rp2.shape[0])
    print("RMSE:", rmse)


def main():
    rp = np.loadtxt("transform_robot.txt")
    cp = np.loadtxt("transform_pointscam.txt")

    affine(rp, cp)
    rigid(rp.T, cp.T)
    


if __name__ == "__main__":
    main()
