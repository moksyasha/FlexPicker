import torch
import argparse
import os
import platform
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import imutils
import socket
import pyrealsense2 as rs
import time

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

from threading import Thread
from threading import Event
import pyzbar.pyzbar as pyzbar

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_PT = ROOT / "best_segm.pt"
stop_thread = False

def get_center(img_orig, model):

    stride, names, pt = model.stride, model.names, model.pt
    
    img = img_orig.transpose((2, 0, 1))[::-1]   # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device).float()
    img /= 255.

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    pred, proto = model(img, augment=False, visualize=False)[:2]

    conf_thres=0.70
    iou_thres=0.45
    classes=None
    agnostic_nms=False
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=100, nm=32)

    annotator = Annotator(img_orig, line_width=3, example=str(names))
    # Process predictions
    det = pred[0]
    
    cX = 0
    cY = 0

    if len(det): # if found smth
        #(det[:, 6:] набор коорд боксов с conf и class shape(n, 32)
        masks = process_mask(proto[0], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC #shape(n, 640, 640)
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_orig.shape).round()  # rescale boxes to im0 size  #shape(n, 38)
        ind_conf = torch.argmax(det[:, 4]) # find index of max confidence

        #Mask plotting
        annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=img[0])
        
        #Write results
        for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
            c = int(cls)  # integer class
            label = f'{names[c]} {conf:.2f}'
            color_bbox = 0 if (j!=ind_conf) else 10
            annotator.box_label(xyxy, label, color=colors(color_bbox, True))

        img = annotator.result()
        
        # find center of area 
        mask = masks[ind_conf].cpu().detach().numpy()
        mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0).astype(np.uint8)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        M = cv2.moments(cnts[0])
        cX = int(M["m10"] / (M["m00"]+1e-10))
        cY = int(M["m01"] / (M["m00"]+1e-10))
	    # draw the contour and center of the shape on the image
        
    img = annotator.result()
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.imshow("a", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    print("Found with center: ", cX, cY)
    return cX, cY



def get_command(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    rp2 = np.dot(trans, cp_new.T)
    rp2[0] -= 50
    rp2[1] -= 20
    rp2[2] += 20

    stri = np.array2string(rp2, formatter={'float_kind':lambda x: "%.2f" % x})
    cmd = "MJ_ARC " + stri[1:-1]
    return cmd, rp2


def cmd_to_robot(sock, cmd):
    sock.send(cmd.encode('ASCII'))
    data = sock.recv(1024)
    print(data.decode('ASCII') + " :" + cmd)


def decodeBar(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)

        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return image


def camera_thread(camera, out, stop_event):
    while True:
        ret, frame = camera.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im = decodeBar(frame)
        cv2.imshow("camera", im)
        out.write(im)
        if stop_event.is_set():
            print("camera thread closed!")
            break


def main():
    # Load tranform matrix
    trans = np.loadtxt("matrix.txt")

    # web camera
    camera = cv2.VideoCapture(3)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (480,640))
    stop_event = Event()
    thread_camera = Thread(target=camera_thread, args=(camera, out, stop_event,))
    
    #create socket, connect to robot controller and send data
    sock = socket.socket()
    sock.connect(("192.168.125.1", 1488))

    #cmd_to_robot(sock, "VALVE_OPEN")
    #cmd_to_robot(sock, "PUMP_STOP ")

    # Load model
    device = select_device(""
    )
    model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((1280, 736), s=stride)  # check image size

    # Run inference
    #model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    #img_orig = cv.imread("3.jpg")

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
    fltr = rs.temporal_filter()

    is_work = 1

    while(is_work):
        cmd_to_robot(sock, "VALVE_OPEN ")
        cmd_to_robot(sock, "PUMP_START ")
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())

        pad = np.ones((16, 1280, 3), dtype=np.uint8)
        color_frame = np.append(color_frame, pad, axis=0)

        x, y = get_center(color_frame, model)

        if x==0:
            print("Didnt find center")
            continue

        # get depth
        while True:

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame or not aligned_frames:
                continue

            pc = rs.pointcloud()
            depth_frame_fltr = fltr.process(depth_frame)
            points_pc = pc.calculate(depth_frame_fltr)
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height
            verts = np.asarray(points_pc.get_vertices()).view(np.float32).reshape(h, w, 3)[:, :, 2]
            
            depth_matr = verts[int(y)-2:int(y)+5, int(x)-2:int(x)+5]
            nonzero = depth_matr[np.nonzero(depth_matr)]

            if nonzero.shape[0] == 0:
                continue
            
            depth = np.mean(nonzero)

            if depth == 0 or np.isnan(depth):
                time.sleep(0.2)
            else:
                break

        cam_points = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                    [x, y], depth))

        if np.isnan(cam_points[0]):
            print("Camera return Nones")
            continue

        #print("Depth: ", depth, "\nCamera points: ", cam_points)
        cmd, coord = get_command(cam_points * 1000, trans)
        #print("\nCommand for robot: ", cmd)

        thread_camera.start()
        # takes a box
        cmd_to_robot(sock, cmd)

        # pick box up
        coord[2] += 100
        stri = np.array2string(coord, formatter={'float_kind':lambda x: "%.2f" % x})
        cmd = "MJ " + stri[1:-1]
        cmd_to_robot(sock, cmd)
        # print("e to exit")
        # a = input()

        # if a == "e":
        #     is_work = 0
        #     break
        # elif a == "c":
        #     continue
        
        #cmd_to_robot(sock, "VALVE_CLOSE ")

        cmd_to_robot(sock, "MJ -165 -228 73")
        cmd_to_robot(sock, "MJ -165 -228 33")

        cmd_to_robot(sock, "ROT ")
        cmd_to_robot(sock, "ROT ")

        cmd_to_robot(sock, "VALVE_CLOSE ")
        cmd_to_robot(sock, "MJ 0 0 200")

        stop_event.set()
        thread_camera.join()
        is_work = 0

    camera.release()
    out.release()
    cv2.destroyAllWindows()
    # cmd_to_robot(sock, "MJ 0 0 200")
    # cmd_to_robot(sock, "VALVE_OPEN ")
    # cmd_to_robot(sock, "PUMP_STOP ")

if __name__ == '__main__':
    main()
