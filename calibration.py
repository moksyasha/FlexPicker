# import cv2 as cv
import socket
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import math
from time import gmtime, strftime
import re
import os


def get_center(image):
    # Применим небольшое размытие для устранения шумов
    image = cv2.GaussianBlur(image, (7, 7), 0)
    # Переведём в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Сегментируем красный цвет
    saturation = hsv[..., 1]
    saturation[(hsv[..., 0] > 15) & (hsv[..., 0] < 165)] = 0
    _, image1 = cv2.threshold(saturation, 92, 255, cv2.THRESH_BINARY)
    mask = image1
    # Найдем наибольшую связную область
    contours = cv2.findContours(image1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    # Оценим ее центр
    b_circle = cv2.minEnclosingCircle(contour)
    b = tuple(int(item) for item in b_circle[0])
    return b, mask


def get_world_coords(u, v, depth):
    camera_matrix = [[386.420, 0.0, 315.6], [0.0, 386.420, 241.429], [0.0, 0.0, 1.0]]
    f = np.linalg.inv(camera_matrix)
    l = np.array([u,v,1]) * depth
    return np.dot(f,l)

def main():
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
    points = np.array(["100 0 0", "-100 0 0", "0 100 0", "0 -100 0",
                       "100 0 100", "-100 0 100", "0 100 100", "0 -100 100", "0 0 50"])
    main_cpoint = []
    main_rpoint = []
    folder = "calibration_" + strftime("%m_%d_%H_%M", gmtime())
    os.mkdir(folder)
    for i in range(9):
        # cmd = input()
        cmd = "MJ " + points[i]
        sock.send(cmd.encode('ASCII'))

        data = sock.recv(1024)
        print(data.decode('ASCII'))

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
        center, mask = get_center(color_image)
        x, y = center
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

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
    print(main_cpoint, main_rpoint)
    transform = cv2.estimateAffine3D(np.array(main_cpoint), np.array(main_rpoint))[1]
    print("Got transform: ", transform)
    np.savetxt('transform.txt', transform)


if __name__ == "__main__":
    main()


