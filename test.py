import numpy as np
import cv2


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


def main():
    rp = np.loadtxt("transform_robot.txt")
    cp = np.loadtxt("transform_pointscam.txt")

    _,trans, p = cv2.estimateAffine3D(rp.T, cp.T, confidence=0.90, ransacThreshold=200)
    print("p: \n", p)

    rp_new = np.append(rp, np.ones(8))
    rp_new = np.resize(rp_new, (4, 8))
    print(rp_new)
    cp2 = trans@rp_new
    print(cp)
    print("new cp: \n:", cp2)

    err = cp2 - cp
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/8)
    print("RMSE:", rmse)

    a = np.array([[1,1], [2,2]])
    b = np.array([[1, 1], [2, 2]])
    # print(R, t)
    # transform = np.append(R, t, axis=1)

    # print("transform:\n\n", transform)

    # camera = np.array([374., 248., 1.803])
    # print("camera: \n", camera)
    # print(camera@R + t.reshape(1, 3))
    #camera = np.append(camera, 1)
    # # print(camera)
    # transform = np.append(transform, np.array([0., 0., 0., 1.]))
    # transform.resize(4, 4)
    # print("new transform: \n\n", transform)

    # result = np.multiply(camera, transform)
    # print("result: \n", result)


if __name__ == "__main__":
    main()
