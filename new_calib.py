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

camera_matrix = [[644.034, 0.0, 632.666], [0.0, 644.034, 362.382], [0.0, 0.0, 1.0]]

def get_center(img):
    #img = cv2.rectangle(img, (0, 0), (1050, 620), (255, 255, 255), 150)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filter1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([0, 255, 255]))
    filter2 = cv2.inRange(img_hsv, np.array([168, 100, 100]), np.array([179, 255, 255]))
    red = cv2.addWeighted(filter1, 1.0, filter2, 1.0, 0.0)

    red = cv2.GaussianBlur(red, (3, 3), 0)
    red = cv2.medianBlur(red, 3)

    # cv2.imshow('RealSense', red)
    # cv2.waitKey(500)

    cv2.destroyAllWindows()
    #circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, red.shape[0] / 8, param1=30, param2=50, minRadius = 1, maxRadius = 100)
    cnts = cv2.findContours(red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) > 2 and area > 70:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (36, 255, 12), 2)
            return int(x), int(y), img

    return 0, 0, 0


def get_world_coords(x, y, depth):
    """return physical coordinates in mm

    Keyword arguments:
    x, y -- coordinates of a point in pixels
    depth -- depth coordiante of the same point
    camera_matrix -- 3x3 matrix with focal lengthes and principial point"""
    f = np.linalg.inv(camera_matrix)
    v = np.array([x, y, 1]) * depth
    return np.dot(f, v)


def main():

    # create socket, connect to RobotStudio server and send data
    sock = socket.socket()
    sock.connect(("192.168.125.1", 1488))

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4) # High density preset

    align_to = rs.stream.depth
    align = rs.align(align_to)

    points = np.array(["300 100 0", "-300 100 0", "-300 -100 0", "300 -100 0",
                        "300 -100 200", "-300 -100 200", "-300 100 200", "300 100 200", "0 0 100"])
                        # "200 200 10", "-200 200 10", "-200 -200 10", "200 -200 10",
                        # "200 -200 200", "-200 -200 200", "-200 200 200", "200 200 200", "0 0 200"])
    
    main_arr = []
    robot_points = []

    # folder = "/calibration_" + strftime("%m_%d_%H_%M_%S", gmtime()) + "/"
    # os.mkdir(folder)

    for i in range(9):
        
        cmd = "MJ " + points[i]
        sock.send(cmd.encode('ASCII'))

        data = sock.recv(1024)
        print(data.decode('ASCII'))

        input()
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())

        if not depth_frame or not aligned_frames:
            continue

        # try https://github.com/IntelRealSense/librealsense/issues/6749
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        x, y, result = get_center(color_frame)
        if x == 0:
            print("Couldnt find center! " + str(i))
            continue

        cv2.imshow("a", result)
        #cv2.imwrite(str(i) + '.png', color_frame)
        cv2.waitKey(1000)

        pc = rs.pointcloud()
        points_pc = pc.calculate(depth_frame)
        verts = np.asarray(points_pc.get_vertices()).view(np.float32).reshape(h, w, 3)

        # save depth channel
        #np.savetxt("depth_" + str(i) + ".txt", verts[:, :, 2])

        dx, dy, dz = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                    [x, y], verts[int(y)][int(x)][2])

        if dz == 0:
            print("Couldnt find depth! " + str(i))
            continue

        distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
        arr = []
        arr.append(dx)
        arr.append(dy)
        arr.append(dz)
        arr.append(distance)
        print("2 Distance from camera to pixel:", distance)
        print("2 Z-depth from camera surface to pixel surface:", dz)
        main_arr.append(arr)

        str_point = np.array(re.findall('-?\d+\.?\d*', points[i])).astype(np.float32)
        robot_points.append(str_point)

    robot_points = np.array(robot_points)
    main_arr = np.array(main_arr)
    np.savetxt("cp.txt", main_arr)
    np.savetxt("rp.txt", robot_points)


if __name__ == "__main__":
    main()
