import torch
import argparse
import os
import platform
import sys

import cv2 as cv
import numpy as np
import imutils
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

    conf_thres=0.85
    iou_thres=0.8
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
        (rbt_pnts_align[:, 2] > -30) & (rbt_pnts_align[:, 2] < 200)
    
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