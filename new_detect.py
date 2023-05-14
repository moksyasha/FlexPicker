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

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
#                            strip_optimizer)
# from utils.plots import Annotator, colors, save_one_box
# from utils.segment.general import masks2segments, process_mask, process_mask_native
# from utils.torch_utils import select_device, smart_inference_mode

from threading import Thread
from threading import Event
import pyzbar.pyzbar as pyzbar
from matplotlib import pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_PT = ROOT / "best_x_50epochs.pt"
stop_thread = False


def get_center(img_orig, model, show_output=0):

    stride, names, pt = model.stride, model.names, model.pt
    
    img = img_orig.transpose((2, 0, 1))[::-1]   # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device).float()
    img /= 255.
    #img[:, :, 640:] = 0
    
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    pred, proto = model(img, augment=False, visualize=False)[:2]

    conf_thres=0.70
    iou_thres=0.45
    classes=None
    agnostic_nms=False
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=100, nm=32)

    # Process predictions
    det = pred[0]
    
    cX = 0
    cY = 0
    angle = 0

    if len(det): # if found smth
        #(det[:, 6:] набор коорд боксов с conf и class shape(n, 32)
        masks = process_mask(proto[0], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC #shape(n, 640, 640)
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_orig.shape).round()  # rescale boxes to im0 size  #shape(n, 38)
        ind_conf = torch.argmax(det[:, 4]) # find index of max confidence

        mask = masks[ind_conf].cpu().detach().numpy()
        mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0).astype(np.uint8)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        largest_cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        rect = cv.minAreaRect(largest_cnt)
        (cX, cY), _, angle = rect

        angle = int(angle)
        if angle > 45:
            angle = -90 + angle 

        if show_output:
            annotator = Annotator(img_orig, line_width=3, example=str(names))
            #Mask plotting
            annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=img[0])

            img_orig = annotator.result()

            #Write results
            for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                #x1, y1, x2, y2 = list(map(lambda x: x.cpu().detach().numpy().astype(int), xyxy))
                #box_img = img_orig[y1-3:y2+3, x1-3:x2+3, :]
                thickness = 10 if j == ind_conf else 2
                mask = masks[j].cpu().detach().numpy()
                mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0).astype(np.uint8)
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                largest_cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
                rect = cv.minAreaRect(largest_cnt)
                (cirX, cirY), _, angle_box = rect
                
                # counter clock wise
                if angle_box > 45:
                    angle_box = -90 + angle_box

                cv2.circle(img_orig, (int(cirX), int(cirY)), 3, (255, 255, 255), -1)
                img_orig = cv2.putText(img_orig, str(int(angle_box)), (int(cirX), int(cirY-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img_orig, [box], 0, (255, 0, 0), thickness)

                # c = int(cls)  # integer class
                # label = f'{names[c]} {conf:.2f}'
                # color_bbox = 0 if (j!=ind_conf) else 10
                # annotator.box_label(xyxy, label, color=colors(color_bbox, True))

            cv.imshow('img', img_orig)
            cv2.waitKey(1000)

    print(f"Found with center: {cX, cY}, angle: {angle}")
    return int(cX), int(cY), angle



def get_command(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    rp2 = np.dot(trans, cp_new.T)
    rp2[0] -= 60
    rp2[1] -= 15
    rp2[2] += 17

    stri = np.array2string(rp2, formatter={'float_kind':lambda x: "%.2f" % x})
    cmd = "MJ_ARC " + stri[1:-1]
    return cmd, rp2


def get_str_command(cam_point):
    stri = np.array2string(cam_point, formatter={'float_kind':lambda x: "%.2f" % x})
    cmd = "MJ_ARC " + stri[1:-1]
    return cmd


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
        # check barcode in database
        if barcodeData in database["code"].values:
            category = 1
            print("found in db")
        else:
            category = 0
            print("not found in db")
        
    return image


def camera_thread(camera, stop_event):
    while True:
        ret, frame = camera.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im = decodeBar(frame)
        cv2.imshow("camera", im)
        cv2.waitKey(10)
        #out.write(im)
        if stop_event.is_set():
            print("camera thread closed!")
            break


def get_img_affine(color_frame, depth_frame_fltr, trans):
    # rotation and translation matrix
    R, t = trans[:, [0, 1, 2]], trans[:, [3]].reshape(3)
    t[[0, 1]] = t[[1, 0]]
    t[2] = -t[2]

    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points_pc = pc.calculate(depth_frame_fltr)
    
    depth_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    verts = np.asarray(points_pc.get_vertices(2)).view(np.float32).reshape(h, w, 3) * 1000
    verts = verts.reshape(-1, 3)
 
    # affine transform camera point to robot system
    rbt = (R@verts.T).T - t #[-580, 115, 790] # -t

    # uv - texture coordinates of points in the pointcloud (0..1, 0..1) normilized by width and height of the color image
    tex = np.asanyarray(points_pc.get_texture_coordinates(2)).reshape(h, w, 2)
    #tex = tex[30:330, 100:950, :] workspace
    tex = tex.reshape(-1, 2)

    color_image = np.asanyarray(color_frame.get_data())

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

    #h = color_image.shape[0]
    #w = color_image.shape[1]
    # criteria for vertices
    cond = (rbt_pnts_align[:, 0] > 0) & (rbt_pnts_align[:, 0] < w) & \
        (rbt_pnts_align[:, 1] > 0) & (rbt_pnts_align[:, 1] < h) & \
        (rbt_pnts_align[:, 2] > -50) & (rbt_pnts_align[:, 2] < 400)
    
    rbt_pnts_align = rbt_pnts_align[cond]
    # get texture for vertices
    q = color_image[tex_align[:, 1], tex_align[:, 0]]

    xt, yt, pt, depth = torch.from_numpy(rbt_pnts_align[:, 0]).to(torch.int64),\
            torch.from_numpy(rbt_pnts_align[:, 1]).to(torch.int64), \
            torch.from_numpy(q[cond]).to(torch.int64), torch.from_numpy(rbt_pnts_align[:, 2]).to(torch.int64)

    newimg = torch.zeros((720, 1280, 3), dtype=torch.int64)
    depth_img = torch.zeros((720, 1280), dtype=torch.int64)

    depth_img.index_put_((yt, xt), depth)
    depth_img.index_put_((yt - 1, xt), depth)
    depth_img.index_put_((yt, xt - 1), depth)
    depth_img.index_put_((yt - 1, xt - 1), depth)

    newimg.index_put_((yt, xt), pt)
    newimg.index_put_((yt - 1, xt), pt)
    newimg.index_put_((yt, xt - 1), pt)
    newimg.index_put_((yt - 1, xt - 1), pt)

    return newimg.numpy().astype(np.uint8)[..., ::-1], depth_img.numpy()


def main():
    global database
    database = pd.read_csv("database.csv")

    global category
    category = 0

    # Load tranform matrix
    trans = np.loadtxt("matrix2.txt")
    R, t = trans[:, [0, 1, 2]], trans[:, [3]].reshape(3)
    # t[[0, 1, 2]] = t[[1, 0, 2]]
    #create socket, connect to robot controller and send data
    #sock = socket.socket()
    #sock.connect(("192.168.125.1", 1488))

    # Load model
    # device = select_device("")
    # model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
    # stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size((1280, 736), s=stride)  # check image size

    # Run inference
    # model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, "2.bag") ## DEL
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_sensor.set_option(rs.option.visual_preset, 4) # High density preset

    align_to = rs.stream.depth
    align = rs.align(align_to)
    fltr = rs.hole_filling_filter(2)

    # # start thread web camera
    # camera = cv2.VideoCapture(3)
    # #out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (480,640))
    # stop_event = Event()
    # thread_camera = Thread(target=camera_thread, args=(camera, stop_event,))
    # thread_camera.start()

    # one box
    is_work = 1
    
    while(is_work):
        
        try:
            #cmd_to_robot(sock, "MJ 0 0 200")
            #cmd_to_robot(sock, "VALVE_OPEN ")
            for x in range(5):
                frames = pipeline.wait_for_frames()
            
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_frame_fltr = fltr.process(depth_frame)

            # convert camera image to robot space
            color_frame_affine, rbt_verts = get_img_affine(color_frame, depth_frame_fltr, trans)

            pad = np.ones((16, 1280, 3), dtype=np.uint8)
            color_frame_affine = np.append(color_frame_affine, pad, axis=0)

            x, y, angle = get_center(color_frame_affine, model, 1)

# самая правая -70 -171 130
# центр 69 -236 92
# 161 -120 97

            # plt.figure(figsize=(21, 9))
            # plt.subplot(131)
            # plt.imshow(rbt_verts)
            # plt.subplot(132)
            # plt.imshow(color_frame_affine)
            # plt.subplot(133)
            # plt.imshow(np.asanyarray(color_frame.get_data()))
            # plt.show()

            # take mean depth of box
            depth_matr = rbt_verts[int(y)-2:int(y)+5, int(x)-2:int(x)+5]
            nonzero = depth_matr[np.nonzero(depth_matr)]

            if nonzero.shape[0] == 0:
                print("1Depth == 0")
                continue
            
            depth = np.mean(nonzero)

            if depth == 0 or np.isnan(depth):
                print("2Depth == 0")
                continue

            # apply translation 
            # check t !
            trans_rbt_pnt = np.array([x, y, depth]) - t
            cmd_box = get_str_command(trans_rbt_pnt)

            # rotate manipulator before taking
            cmd_to_robot(sock, "ROT " + str(-angle))

            # take a box
            cmd_to_robot(sock, "PUMP_START ") ## не помню как брать
            cmd_to_robot(sock, cmd_box)
            time.sleep(1)

            # pick box up
            cmd_box[2] = 250
            stri = np.array2string(coord, formatter={'float_kind':lambda x: "%.2f" % x})
            cmd = "MJ " + stri[1:-1]
            cmd_to_robot(sock, cmd)

            # rotate manipulator after taking for alignment
            cmd_to_robot(sock, "ROT " + str(angle))

            # moving to camera for detecting !! изменить координаты камеры
            cmd_to_robot(sock, "MJ -165 -228 180")
            cmd_to_robot(sock, "MJ -165 -228 33")

            # detecting qr
            for i in range(1):
                time.sleep(1.5)
                if category:
                    break
                cmd_to_robot(sock, "ROT 90")
                time.sleep(1.5)
                if category:
                    break
                cmd_to_robot(sock, "ROT 90")
                time.sleep(1.5)
                if category:
                    break
                cmd_to_robot(sock, "ROT 90")
                time.sleep(1.5)

            # TODO category

            cmd_to_robot(sock, "VALVE_CLOSE ")
            cmd_to_robot(sock, "PUMP_STOP ") ## не помню как отпускать
            time.sleep(2)
            cmd_to_robot(sock, "VALVE_OPEN ")
            cmd_to_robot(sock, "ROT_BASE ")

            is_work -= 1

        except Exception as e:
            print("Something went wrong")
            print(e)
            is_work = 0

    stop_event.set()
    thread_camera.join()
    cmd_to_robot(sock, "PUMP_STOP ")
    camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
