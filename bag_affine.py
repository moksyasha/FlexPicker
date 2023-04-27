import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal
from math import ceil
import torchvision.transforms as T



def get_robot(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    return np.dot(trans, cp_new.T)


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


def get_robot(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    return np.dot(trans, cp_new.T)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, "1.bag")
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    try:

        trans = np.loadtxt("matrix.txt")
        R, t = trans[:, [0, 1, 2]], trans[:, [3]].reshape(3)
        print(t)
        t[[0, 1]] = t[[1, 0]]
        print(t)
        align_to = rs.stream.depth
        align = rs.align(align_to)
        fltr = rs.hole_filling_filter(2)

        profile = pipeline.start(config)

        while True:
            # 100 30
            # 950 330
            for x in range(5):
                frames = pipeline.wait_for_frames()
            
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            # depth_image = np.asanyarray(depth_frame.get_data())/1000.
            # print("depth : ", depth_image.shape, depth_image[200])
                # mask = np.where(depth == 0, 1, 0).astype('uint8')
            #depth = cv.inpaint(depth, mask, 3, flags=cv.INPAINT_NS)
            color_image = np.asanyarray(color_frame.get_data())
            # take only points in workspace
            color_image_crop = color_image[30:330, 100:950, :]
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            depth_frame_fltr = fltr.process(depth_frame)

            points_pc = pc.calculate(depth_frame_fltr)
            
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height
            verts = np.asarray(points_pc.get_vertices(2)).view(np.float32).reshape(h, w, 3)*[1000, 1000, 1000]
            # take only points in workspace
            print(verts.shape)
            verts = verts[30:330, 100:950, :]
            w_dpth, h_dpth = verts.shape[1], verts.shape[0]
            verts = verts.reshape(-1, 3)

            # affine transform camera point to robot system
            rbt = (R@verts.T).T - t

            #robot_points = robot_points + t

            # uv - texture coordinates of points in the pointcloud (0..1, 0..1) normilized by width and height of the color image
            tex = np.asanyarray(points_pc.get_texture_coordinates(2)).reshape(h, w, 2)
            tex = tex[30:330, 100:950, :]
            tex = tex.reshape(-1, 2)
            h = color_image.shape[0]
            w = color_image.shape[1]
            
            tex[:, 0] = tex[:, 0] * w
            tex[:, 1] = tex[:, 1] * h
            tex = tex.astype("uint16")

            # sort from highest point to lowest
            ind = np.argsort(rbt[:, 2])[::-1]

            # align texture and robot 3d points
            rbt_pnts_align = rbt[ind]#.astype(np.int16)
            tex_align = tex[ind]

            h = color_image_crop.shape[0]
            w = color_image_crop.shape[1]
            cond = (rbt_pnts_align[:, 0] > 0) & (rbt_pnts_align[:, 0] < w) & \
                (rbt_pnts_align[:, 1] > 0) & (rbt_pnts_align[:, 1] < h) & (rbt_pnts_align[:, 2] != 0)
            
            rbt_pnts_align = rbt_pnts_align[cond]

            q = color_image[tex_align[:, 1], tex_align[:, 0]]

            xt, yt, pt = torch.from_numpy(rbt_pnts_align[:, 0]).to(torch.int64),\
                    torch.from_numpy(rbt_pnts_align[:, 1]).to(torch.int64), \
                    torch.from_numpy(q[cond]).to(torch.int64)

            newimg = torch.zeros((h_dpth, w_dpth, 3), dtype=torch.int64)
            newimg.index_put_((yt, xt), pt)#, accumulate=True)

            transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 0.5))
            blurred_img = transform(newimg)

            plt.figure(figsize=(18, 9))
            plt.subplot(121)
            plt.imshow(newimg)
            plt.subplot(122)
            plt.imshow(color_image_crop)
            plt.show()
            break
            
    except Exception as e:
        print(e)
        pipeline.stop()

if __name__ == '__main__':
    main()