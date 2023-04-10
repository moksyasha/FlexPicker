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
import pyrealsense2 as rs

def main():

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
        
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())

    pad = np.ones((16, 1280, 3), dtype=np.uint8)
    color_frame = np.append(color_frame, pad, axis=0)

    cv.imwrite("4.jpg", color_frame)
    depth_frame = aligned_frames.get_depth_frame()

    pc = rs.pointcloud()
    depth_frame_fltr = fltr.process(depth_frame)
    points_pc = pc.calculate(depth_frame_fltr)
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    verts = np.asarray(points_pc.get_vertices()).view(np.float32).reshape(h, w, 3)[:, :, 2]
    print(verts.shape)
    np.savetxt("4.txt", verts)


    

    

if __name__ == '__main__':
    main()
