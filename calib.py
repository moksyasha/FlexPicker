# import cv2 as cv
import socket
import time
#import pyrealsense2 as rs
import numpy as np
import cv2
import math
from time import gmtime, strftime
import re
import os
import pyrealsense2 as rs
 
# https://solarianprogrammer.com/2015/05/08/detect-red-circles-image-using-opencv/
def get_center(img):
    img = cv2.rectangle(img, (0, 0), (640, 420), (255, 255, 255), 180)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filter1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([0, 255, 255]))
    filter2 = cv2.inRange(img_hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
    red = cv2.addWeighted(filter1, 1.0, filter2, 1.0, 0.0)

    red = cv2.GaussianBlur(red, (3, 3), 0)
    red = cv2.medianBlur(red, 3)

    cv2.imshow('RealSense', red)
    cv2.waitKey(500)

    cv2.destroyAllWindows()
    #circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, red.shape[0] / 8, param1=30, param2=50, minRadius = 1, maxRadius = 100)
    cnts = cv2.findContours(red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) > 2 and area > 20 and area < 120:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (36, 255, 12), 2)
            return int(x), int(y), img

    return 0, 0, 0


def main():

    # img = cv2.imread('2.png', cv2.IMREAD_COLOR)[0:480, 0:640]
    # cv2.imshow('frame1', get_center(img)[2])
    # cv2.waitKey(0)

    # create socket, connect to RobotStudio server and send data
    sock = socket.socket()
    sock.connect(("192.168.125.1", 1488))

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start streaming
    pipeline.start(config)
    points = np.array(["200 50 0", "-200 50 0", "-200 -50 0", "200 -50 0",
                        "200 50 100", "-200 50 100", "-200 -50 100", "200 -50 100"])
    #points = np.array(["-300 100 0", "-300 -100 0", "300 -100 0", "300 100 0", "0 0 0"])
    main_cpoint = []
    main_rpoint = []
    center = []
    #folder = "calibration_" + strftime("%m_%d_%H_%M_%S", gmtime())
    #os.mkdir(folder)

    # depth_intrinsic = rs.intrinsics()
    # depth_intrinsic.width = 640
    # depth_intrinsic.height = 480
    # depth_intrinsic.ppx = 315.599761
    # depth_intrinsic.ppy = 241.429473
    # depth_intrinsic.fx = 386.4204406
    # depth_intrinsic.fy = 386.4204406
    # depth_intrinsic.model = rs.distortion.brown_conrady
    # depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    # print(depth_intrinsic)
    ok_dots = 0
    for i in range(9):
        # cmd = input()
        cmd = "MJ " + points[i]
        sock.send(cmd.encode('ASCII'))

        data = sock.recv(1024)
        print(data.decode('ASCII'))
        time.sleep(3)
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        #rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], depth)
        # first try - change color_intin

        # coef in https://github.com/IntelRealSense/librealsense/issues/3569#issuecomment-475896784

        # 2 try
        # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # 3 try kudinov
        depth_intrinsic = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        print(depth_intrinsic)
        # in addition - change depth to this
        # self.verts[:] = np.asarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)
        # points = pc.calculate(depth_frame)


        # Convert images to numpy arrays
        #color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        x, y, result = get_center(color_image)
        ok_dots += 1
        if not x:
            ok_dots-=1
            print("Couldnt find center! Skip point: ", i)
            continue

        cv2.imshow('RealSense', result)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        depth = depth_frame.get_distance(x, y)

        for j in range(10):
            dx, dy, dz = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [x, y], depth)
            if dx != 0:
                break

        if dx == 0:
            ok_dots-=1
            print("Error! dx == 0, point: ", i)
            continue
        # dx, dy, depth = np.asarray(rs.rs2_deproject_pixel_to_point(
        #             depth_intrinsic, [closest_coords[0], closest_coords[1]], 
        #             self.verts[closest_coords[1]][closest_coords[0]][2]))

        distance = math.sqrt(((dx) ** 2) + ((dy) ** 2) + ((dz) ** 2))
        print("Distance from camera to pixel:", distance, "x, y: ", dx, dy)
        print("Z-depth from camera surface to pixel surface:", depth)
        cam_coord = np.array([dx*1000, dy*1000, depth*1000])
        main_cpoint.append(cam_coord)
        str_point = np.array(re.findall('-?\d+\.?\d*', points[i])).astype(np.float32)
        main_rpoint.append(str_point)
        center.append([x, y])
        # img_name = '\\' + folder + '\\' + str(i) + '.png'
        cv2.imwrite(str(i) + '.png', color_image)
        # cv2.imwrite(str(i) + '.png', images)

    #print("POINTS:", main_cpoint, main_cpoint.shape, main_rpoint, main_rpoint.shape)

    #main_cpoint = np.rot90(main_cpoint, k=-1)
    #main_rpoint = np.rot90(main_rpoint, k=-1)
    main_cpoint = np.array(main_cpoint)
    main_rpoint = np.array(main_rpoint)
    center_ar = np.array(center)
    # _,trans, p = cv2.estimateAffine3D(main_cpoint, main_rpoint, confidence=0.90, ransacThreshold=200)
    # print("p: \n", p)

    # cp_new = np.append(main_cpoint, np.ones(ok_dots))
    # cp_new = np.resize(cp_new, (4, ok_dots))

    # rp2 = trans@cp_new
    # err = rp2 - rp
    # err = err * err
    # err = np.sum(err)
    # rmse = np.sqrt(err/ok_dots)
    # print("RMSE:", rmse)

    np.savetxt('transform_pointscam.txt', main_cpoint)
    np.savetxt('transform_robot.txt', main_rpoint)
    np.savetxt('center.txt', center)

if __name__ == "__main__":
    main()
