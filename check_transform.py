import numpy as np
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

camera_matrix = [[644.034, 0.0, 632.666], [0.0, 644.034, 362.382], [0.0, 0.0, 1.0]]

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
    _,trans, p = cv2.estimateAffine3D(cp, rp)
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
    return(trans)

def rigid(rp, cp):
    print("Orig: \n", np.array(rp).astype(np.int32))
    ret_R, ret_t = rigid_transform_3D(cp, rp)

    cpnew = np.append(cp, np.ones((1, cp.shape[1])), axis=0)
    matrix = np.append(ret_R, ret_t, axis=1)
    print("Matrix: \n", matrix)
    rp3 = matrix@cpnew
    print("new: \n", rp3.astype(np.int32))
    err = rp3 - rp
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/9)
    print("RMSE:", rmse)
    return matrix


def get_world_coords(x, y, depth):
    f = np.linalg.inv(camera_matrix)
    v = np.array([x, y, 1]) * depth
    return np.dot(f, v)

def make_intrinsics():
    intrinsics = rs.intrinsics()
    intrinsics.coeffs = [0,0,0,0,0]
    intrinsics.fx = 644.034
    intrinsics.fy = 644.034
    intrinsics.height = 720
    intrinsics.ppx = 632.666
    intrinsics.ppy = 362.382
    intrinsics.width=1280
    return intrinsics


def main():
    trans = np.loadtxt("matrix.txt")
    rp = np.loadtxt("rp.txt")
    cp = np.loadtxt("cp.txt")[:, [0, 1, 4]]

    instr = make_intrinsics()
    new = []

    for cx, cy, cd in cp:
        a = list(rs.rs2_deproject_pixel_to_point(instr,
                    [cx, cy], cd))
        new.append(a)
    
    new = np.array(new)
    print(new)
    new = np.hstack((new*1000, np.ones((new.shape[0], 1))))
    print(new)

    new1 = []
    for c in new:
        c_new = (trans@c.T).astype("int32")
        new1.append(c_new)

    print(new1)

    # cp = np.loadtxt("cp.txt") * 1000

    # # # #create 3d axes
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    
    # # ax.plot3D(rp[:, 0], rp[:, 1], rp[:, 2], 'red')
    # # ax.view_init(60, 50)
    # # ax.plot3D(cp[:, 2], cp[:, 3], cp[:, 4], 'blue')
    # # ax.view_init(60, 50)
    # # plt.show()
    # cp = cp[:, [2, 3, 4]]
    # # # # print(rp.shape, cp.shape)
    # trans = affine(rp, cp)
    
    #matrix = rigid(rp.T, cp.T)
    # np.savetxt("matrix_old.txt", matrix)
    


if __name__ == "__main__":
    main()
