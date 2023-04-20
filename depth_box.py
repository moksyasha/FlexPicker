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

from matplotlib import pyplot as plt

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
#                            strip_optimizer)
# from utils.plots import Annotator, colors, save_one_box
# from utils.segment.general import masks2segments, process_mask, process_mask_native
# from utils.torch_utils import select_device, smart_inference_mode

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# PATH_PT = ROOT / "best_segm.pt"


# def get_center(img_orig, model, depth, show_output=0):

#     stride, names, pt = model.stride, model.names, model.pt
    
#     img = img_orig.transpose((2, 0, 1))[::-1]   # HWC to CHW, BGR to RGB
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(model.device).float()
#     img /= 255.
#     #img[:, :, 690:] = 0
    
#     if len(img.shape) == 3:
#         img = img[None]  # expand for batch dim

#     pred, proto = model(img, augment=False, visualize=False)[:2]

#     conf_thres=0.8
#     iou_thres=0.45
#     classes=None
#     agnostic_nms=False
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=100, nm=32)

#     # Process predictions
#     det = pred[0]
    
#     cX = 0
#     cY = 0
#     angle = 0

#     if len(det): # if found smth
#         #(det[:, 6:] набор коорд боксов с conf и class shape(n, 32)
#         masks = process_mask(proto[0], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC #shape(n, 640, 640)
#         det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_orig.shape).round()  # rescale boxes to im0 size  #shape(n, 38)
#         ind_conf = torch.argmax(det[:, 4]) # find index of max confidence

#         if show_output:
#             annotator = Annotator(img_orig, line_width=3, example=str(names))
#             #Mask plotting
#             annotator.masks(
#                     masks,
#                     colors=[colors(x, True) for x in det[:, 5]],
#                     im_gpu=img[0])

#             img_orig = annotator.result()

#             #Write results
#             for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):

#                 mask = masks[j].cpu().detach().numpy().astype(np.uint8)
#                 # mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0).astype(np.uint8)
#                 # https://stackoverflow.com/questions/62370500/get-boolean-array-indicating-which-elements-in-array-which-belong-to-a-list

#                 depth_mask = np.where(mask, depth, 0)
#                 np.savetxt("./depth/mask" + str(j) + ".txt", depth_mask)

#                 c = int(cls)  # integer class
#                 label = f'{names[c]} {conf:.2f}'
#                 color_bbox = 0 if (j!=ind_conf) else 10
#                 annotator.box_label(xyxy, label, color=colors(color_bbox, True))

#             cv.imshow('img', img_orig)
#             cv2.waitKey(0)

#     print(f"Found with center: {cX, cY}, angle: {angle}")
#     return cX, cY, angle


def get_robot(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    return np.dot(trans, cp_new.T)


def main():
    trans = np.loadtxt("matrix.txt")
    depth = np.loadtxt("./depth/5.txt")[100:400, 200:1000]
    print(depth.shape)
    img = cv.imread("./depth/5.jpg")
    x = np.arange(0, 800, 1)
    y = np.arange(0, 300, 1)



    zro = np.zeros([1000, 1000])
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            x, y, d = get_robot(np.array([j, i, depth[i, j]]), trans)
            zro[int(y), int(x)] = d

    plt.matshow(zro, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.show()
    # device = select_device("")
    # model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
    # stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size((1280, 736), s=stride)  # check image size
    # img = cv.imread("./depth/5.jpg")

    # pad = np.ones((16, 1280), dtype=np.uint8)
    # depth = np.append(depth, pad, axis=0)
    # get_center(img, model, depth, 1)


if __name__ == '__main__':
    main()
