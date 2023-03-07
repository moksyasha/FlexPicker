#import cv2 as cv
import socket
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import math
from time import gmtime, strftime
import re
import os


def get_center(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.rectangle(img_hsv, (0, 0), (640, 480), (255, 255, 255), 180)
    filter1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([1, 255, 255]))
    filter2 = cv2.inRange(img_hsv, np.array([167, 100, 100]), np.array([172, 255, 255]))
    red = cv2.addWeighted(filter1, 1.0, filter2, 1.0, 0.0)

    red = cv2.GaussianBlur(red, (3, 3), 0)
    red = cv2.medianBlur(red, 3)

    cv2.imshow('RealSense', img_hsv)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()
    # circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, red.shape[0] / 8, param1=30, param2=50, minRadius = 1, maxRadius = 100)
    cnts = cv2.findContours(red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) > 4  and area > 50 and area < 140:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (36, 255, 12), 2)
            return int(x), int(y), img

    return 0, 0, 0

def get_world_coords(u, v, depth):
    camera_matrix = [[386.420, 0.0, 315.6], [0.0, 386.420, 241.429], [0.0, 0.0, 1.0]]
    f = np.linalg.inv(camera_matrix)
    l = np.array([u,v,1]) * depth
    return np.dot(f,l)


def main():

    transform = np.loadtxt("transform.txt")

    # create socket, connect to RobotStudio server and send data
    #sock = socket.socket()
    #sock.connect(("192.168.125.1", 1488))
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start streaming
    pipeline.start(config)
    points = np.array(["100 100 100", "-100 0 0", "0 100 0", "0 -100 0",
                       "100 0 100", "-100 0 100", "0 100 100", "0 -100 100", "0 0 50"])

    for i in range(1):
        #cmd = input()
        cmd = "MJ " + points[i]
        #sock.send(cmd.encode('ASCII'))
        
        #data = sock.recv(1024)
        #print(data.decode('ASCII'))

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
        #images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', result)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        depth = depth_frame.get_distance(x, y)

        dx, dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], depth)
        distance = math.sqrt(((dx) ** 2) + ((dy) ** 2) + ((dz) ** 2))
        print("Distance from camera to pixel:", distance, "x, y: ", x, y)
        print("Z-depth from camera surface to pixel surface:", depth)
        point = np.array([x, y, depth])

        cam_coords = get_world_coords(x, y, depth)
        print(cam_coords)
        print(point)

if __name__ == "__main__":
    main()


