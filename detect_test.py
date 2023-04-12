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

def get_center(img_orig, model, show_output=0):

    stride, names, pt = model.stride, model.names, model.pt
    
    img = img_orig.transpose((2, 0, 1))[::-1]   # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device).float()
    img /= 255.
    #img[:, :, 690:] = 0
    
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    pred, proto = model(img, augment=False, visualize=False)[:2]

    conf_thres=0.8
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

        if angle > 45:
            angle = -90 + angle

        if show_output:
            annotator = Annotator(img_orig, line_width=3, example=str(names))
            #Mask plotting
            annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=img[0])

            

            #Write results
            for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                #x1, y1, x2, y2 = list(map(lambda x: x.cpu().detach().numpy().astype(int), xyxy))
                #box_img = img_orig[y1-3:y2+3, x1-3:x2+3, :]
                # thickness = 8 if j == ind_conf else 2
                # mask = masks[j].cpu().detach().numpy()
                # mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0).astype(np.uint8)
                # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # cnts = imutils.grab_contours(cnts)
                # largest_cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
                # rect = cv.minAreaRect(largest_cnt)
                # (cirX, cirY), _, angle_box = rect
                
                # # counter clock wise
                # if angle_box > 45:
                #     angle_box = -90 + angle_box
                # cv2.circle(img_orig, (int(cirX), int(cirY)), 3, (255, 255, 255), -1)
                # img_orig = cv2.putText(img_orig, str(int(angle_box)), (int(cirX), int(cirY-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.6, (255, 255, 255), 2, cv2.LINE_AA)
                # box = cv.boxPoints(rect)
                # box = np.int0(box)
                # cv.drawContours(img_orig, [box], 0, (255, 0, 0), thickness)

                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                color_bbox = 0 if (j!=ind_conf) else 10
                annotator.box_label(xyxy, label, color=colors(color_bbox, True))

            img_orig = annotator.result()
            cv.imshow('img', img_orig)
            cv.imwrite('ximg2.jpg', img_orig)
            cv2.waitKey(0)

    print(f"Found with center: {cX, cY}, angle: {angle}")
    return cX, cY, angle
        #     x1, y1, x2, y2 = list(map(lambda x: x.cpu().detach().numpy().astype(int), xyxy))
        #     box = img_orig[y1-3:y2+3, x1-3:x2+3, :]

        #     cv2.imshow("a", box)
        #     cv2.waitKey(0)
            # c = int(cls)  # integer class
            # label = f'{names[c]} {conf:.2f}'
            # color_bbox = 0 if (j!=ind_conf) else 10
            # #x1, y1, x2, y2 = 
            # annotator.box_label(xyxy, label, color=colors(color_bbox, True))

        # img = annotator.result()

        # find center of area
        # for i in masks:

        
        # mask = masks[ind_conf].cpu().detach().numpy()
        # mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0).astype(np.uint8)
        # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # M = cv2.moments(cnts[0])
        # cX = int(M["m10"] / (M["m00"]+1e-10))
        # cY = int(M["m01"] / (M["m00"]+1e-10))
	    # draw the contour and center of the shape on the image
        
    # img = annotator.result()
    # cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    # cv2.imshow("a", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print("Found with center: ", cX, cY)
    # return cX, cY


def get_command(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    rp2 = np.dot(trans, cp_new.T)
    rp2[0] -= 50
    rp2[1] -= 20
    rp2[2] += 20

    stri = np.array2string(rp2, formatter={'float_kind':lambda x: "%.2f" % x})
    cmd = "MJ_ARC " + stri[1:-1]
    return cmd, rp2


def main():

    device = select_device("")
    model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((1280, 736), s=stride)  # check image size
    img = cv.imread("./depth/3.jpg")
    # cv2.imshow("orig", img)
    # cv2.waitKey()

    #pad = np.ones((16, 1280, 3), dtype=np.uint8)
    #img = np.append(img, pad, axis=0)
    get_center(img, model, 1)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()