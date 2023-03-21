import torch
import argparse
import os
import platform
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import imutils

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_PT = ROOT / "best_segm.pt"


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


def get_command(x, y):
    
def main():
    # Load model
    device = select_device("")
    model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((1280, 736), s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    #img_orig = cv.imread("3.jpg")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())

    print(frame.shape) 
    x, y = get_center(color_frame, model)

    sock = socket.socket()
    sock.connect(("192.168.125.1", 1488))

    get_command(x, y)
    sock.send(cmd.encode('ASCII'))

    data = sock.recv(1024)
    print(data.decode('ASCII'))
    input()

    

if __name__ == '__main__':
    main()
