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

# https://solarianprogrammer.com/2015/05/08/detect-red-circles-image-using-opencv/
def get_center(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filter1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([7, 255, 255]))
    filter2 = cv2.inRange(img_hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
    red = cv2.addWeighted(filter1, 1.0, filter2, 1.0, 0.0)

    red = cv2.GaussianBlur(red, (3, 3), 0)
    red = cv2.medianBlur(red, 7)
    
    #circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, red.shape[0] / 8, param1=30, param2=50, minRadius = 1, maxRadius = 100)
    cnts = cv2.findContours(red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) > 3 and area > 20 and area < 140:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (36, 255, 12), 2)
            return x, y, img

    return 0, 0, 0


def get_world_coords(u, v, depth):
    camera_matrix = [[386.420, 0.0, 315.6], [0.0, 386.420, 241.429], [0.0, 0.0, 1.0]]
    f = np.linalg.inv(camera_matrix)
    l = np.array([u,v,1]) * depth
    return np.dot(f,l)


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
    # points = np.array(["100 0 0", "-100 0 0", "0 100 0", "0 -100 0",
    #                    "100 0 100", "-100 0 100", "0 100 100", "0 -100 100", "0 0 50"])
    points = np.array(["-300 100 0", "-300 -100 0", "300 -100 0", "300 100 0"])
    main_cpoint = []
    main_rpoint = []
    folder = "calibration_" + strftime("%m_%d_%H_%M", gmtime())
    os.mkdir(folder)

    for i in range(4):
        # cmd = input()
        cmd = "MJ " + points[i]
        sock.send(cmd.encode('ASCII'))

        data = sock.recv(1024)
        print(data.decode('ASCII'))
        input()

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        x, y, result = get_center(color_image)
        if not x:
            print("Couldnt find center! Retry!")

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((result, depth_colormap))

        depth = depth_frame.get_distance(x, y)

        dx, dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], depth)
        distance = math.sqrt(((dx) ** 2) + ((dy) ** 2) + ((dz) ** 2))
        print("Distance from camera to pixel:", distance)
        print("Z-depth from camera surface to pixel surface:", depth)
        point = np.array([x, y, depth]).astype(np.float32)
        cam_coord = get_world_coords(x, y, depth)
        main_cpoint.append(cam_coord)
        str_point = np.array(re.findall('\d+', points[i])).astype(np.float32)
        main_rpoint.append(str_point)

        img_name = '\\' + folder + '\\' + str(i) + '.png'
        cv2.imwrite(img_name, images)
        cv2.imwrite(str(i) + '.png', images)

    print(main_cpoint, main_rpoint)
    transform = cv2.estimateAffine3D(np.array(main_cpoint), np.array(main_rpoint))[1]
    print("Got transform: ", transform)
    np.savetxt('transform.txt', transform)


if __name__ == "__main__":
    main()
