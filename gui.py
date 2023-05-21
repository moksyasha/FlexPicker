import image_processing as img_proc

import cv2 as cv
import numpy as np
import socket
import sys
import pyrealsense2 as rs
import pandas as pd
import time

from PyQt6.QtCore import QSize, Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtWidgets import QWidget, QLabel, QGridLayout, QApplication, QMainWindow, QPushButton
from PyQt6 import QtGui
from PyQt6.QtGui import QPixmap

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_PT = ROOT / "best_segm.pt"
stop_thread = False


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = cv.VideoCapture("1.mp4")
        self.database = pd.read_csv("database.csv")
        self.cat = 0
        # self.cap.set(cv.CAP_PROP_FPS, 30)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

    def run(self):
        # capture from web cam
        while self._run_flag:
            ret, img = self.cap.read()
            if ret:
                dec = self.decodeBar(img)
                self.change_pixmap_signal.emit(dec)
            else:
                self.cap = cv.VideoCapture("1.mp4")
        self.cap.release()

    def stop(self):
        # waits for thread to finish
        print("Stop camera thread!")
        self._run_flag = False
        self.wait()

    def decodeBar(self, image):
        barcode_detector = cv.barcode_BarcodeDetector()
        # 'retval' is boolean mentioning whether barcode has been detected or not
        retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(image)
        # proceed further only if at least one barcode is detected:
        if retval:
            points = points.astype(np.int32)
            for i, point in enumerate(points):
                if decoded_info[i] == "":
                    continue

                image = cv.drawContours(image, [point], 0, (0, 255, 0), 2)

                x1, y1 = point[1]
                y1 = y1 - 10

                data = int(decoded_info[i])
                if data in self.database['code'].values:
                    name = self.database.loc[self.database['code'] == data]['name']
                    self.cat = 1
                    print(decoded_info[i] + " found in db: " + name)
                else:
                    print(decoded_info[i] + " not found in db")
                cv.putText(image, decoded_info[i], (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, 2)
        return image

    def get_cat(self):
        return self.cat


class RobotThread(QThread):
    yolo_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_tr):
        self.matrix = np.loadtxt("matrix_rigid.txt")
        self.is_work = True
        self.thread_work = True
        self.cur_category = 0
        self.camera_tr = camera_tr

        super().__init__()
        # #create socket, connect to robot controller and send data
        #self.sock = socket.socket()
        #self.sock.connect(("192.168.125.1", 1488))
        # self.sock = socket.socket()
        # self.sock.connect(("127.0.0.1", 1488))

        # # Load model
        device = select_device("")
        self.model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
        self.predict_yolo = np.array([])
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((1280, 736), s=stride)  # check image size

        # # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, "1.bag")
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_sensor.set_option(rs.option.visual_preset, 4) # High density preset

        align_to = rs.stream.depth
        self.align = rs.align(align_to)
        self.fltr = rs.hole_filling_filter(2)

    def get_str_command(self, cam_point):
        stri = np.array2string(cam_point, formatter={'float_kind':lambda x: "%.2f" % x})
        cmd = "MJ " + stri[1:-1]
        return cmd

    def cmd_to_robot(self, sock, cmd):
        sock.send(cmd.encode('ASCII'))
        data = sock.recv(1024)
        print(data.decode('ASCII') + " :" + cmd)
        return

    # returns x, y, angle, pred_image
    def yolo_coord(self):
        # self.cmd_to_robot(self.sock, "MJ 0 0 200")

        #while True:
        frames = self.pipeline.wait_for_frames()
                
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth_frame_fltr = self.fltr.process(depth_frame)

        # convert camera image to robot space
        color_frame_affine, rbt_verts = img_proc.get_img_affine(color_frame, depth_frame_fltr, self.matrix)

        pad = np.ones((16, 1280, 3), dtype=np.uint8)
        color_frame_affine = np.append(color_frame_affine, pad, axis=0)
        x, y, angle, yolo_pred = img_proc.get_center(color_frame_affine, self.model)
        self.yolo_signal.emit(yolo_pred)

        if x == 0:
            print("Didnt find any object!")
        # take mean depth of box
        depth_matr = rbt_verts[int(y)-5:int(y)+5, int(x)-5:int(x)+5]
        nonzero = depth_matr[np.nonzero(depth_matr)]

        if nonzero.shape[0] == 0:
            print("1Depth == 0")
        
        depth = np.mean(nonzero)

        if depth == 0 or np.isnan(depth):
            print("2Depth == 0")

        return x, y, depth, angle, yolo_pred
    
    def get_up_box(self, x, y, depth, angle):
        # apply translation 
        # check t !
        new_t = np.array([-455, -459, 0])
        trans_rbt_pnt = np.array([x, y, depth]) + new_t
        
        trans_rbt_pnt_plus = trans_rbt_pnt
        trans_rbt_pnt_plus[2] = 200

        cmd_box_above = self.get_str_command(trans_rbt_pnt_plus)
        cmd_box = self.get_str_command(trans_rbt_pnt)

        # # # rotate manipulator before taking
        # self.cmd_to_robot(self.sock, "ROT " + str(angle))

        # take a box
        # self.cmd_to_robot(self.sock, cmd_box_above)
        # self.cmd_to_robot(self.sock, "PUMP_START ")
        # self.cmd_to_robot(self.sock, cmd_box)
        time.sleep(1.5)

        # pick box up
        # self.cmd_to_robot(self.sock, cmd_box_above)

        # rotate manipulator after taking for alignment
        # self.cmd_to_robot(self.sock, "ROT " + str(-angle))
        return

    def rotating(self, camera_thread):
        # detecting qr
        for i in range(1):
            time.sleep(1.5)
            cat = self.camera_tr.get_cat()
            if cat!=0:
                break
            # self.cmd_to_robot(self.sock, "ROT 90")
            time.sleep(1.5)
            cat = self.camera_tr.get_cat()
            if cat!=0:
                break
            # self.cmd_to_robot(self.sock, "ROT 90")
            time.sleep(1.5)
            cat = self.camera_tr.get_cat()
            if cat!=0:
                break
            # self.cmd_to_robot(self.sock, "ROT 90")
            time.sleep(1.5)
            cat = self.camera_tr.get_cat()

        self.cur_category = cat
        print(cat)
        return

    def decider(self):
        # final coord boxes for dropping
        if self.cur_category == 1:
            pass#self.cmd_to_robot(self.sock, "MJ -302 -200 60") # найдено в бд
        else:
            print("not found")

        # default values
        # self.cmd_to_robot(self.sock, "8VALVE_1 ") # open value
        # self.cmd_to_robot(self.sock, "PUMP_STOP ")
        # self.cmd_to_robot(self.sock, "8VALVE_0 ") # close default
        # self.cmd_to_robot(self.sock, "ROT_BASE ")
        return

    def run(self):
        print("Sorting...")
        while self.thread_work:

            while self.is_work:
                print(".")
                try:
                    self.cur_category = 0

                    x, y, angle, depth, yolo_pred = self.yolo_coord()

                    if self.is_work==0: break

                    if x == 0 or depth == 0:
                        break
                    
                    self.get_up_box(x, y, depth, angle)

                    if self.is_work==0: break

                    # moving to camera for detecting !! изменить координаты камеры
                    if self.is_work==0: break
                    #self.cmd_to_robot(self.sock, "MJ -302 -478 250")

                    if self.is_work==0: break
                    self.rotating(camera_thread)

                    if self.is_work==0: break
                    self.decider()
                except Exception as e:
                    self.is_work = False
                    print("Something went wrong")
                    print(e)

            # Default values
            print("Sorting finished")
            # self.cmd_to_robot(self.sock, "MJ 0 0 250")
            # self.cmd_to_robot(self.sock, "8VALVE_1 ") # open value
            # self.cmd_to_robot(self.sock, "PUMP_STOP ")
            # self.cmd_to_robot(self.sock, "8VALVE_0 ") # close default
            # self.cmd_to_robot(self.sock, "ROT_BASE ")
        return

    def stop_sort(self):
        # waits for thread to finish
        print("Stop sorting!")
        self.is_work = False
        #self.wait()
    
    def stop(self):
        # waits for thread to finish
        print("Stop thread sorting!")
        self.thread_work = False
        self.is_work = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FlexPicker")
        self.setFixedSize(QSize(1400, 520))

        self.start = QPushButton("start")
        self.start.setCheckable(True)
        self.start.clicked.connect(self.start_clicked)

        self.stop = QPushButton("stop")
        self.stop.setCheckable(True)
        self.stop.clicked.connect(self.stop_clicked)

        self.quit = QPushButton("quit")
        self.quit.setCheckable(True)
        self.quit.clicked.connect(self.quit_clicked)

        self.im_cam_label = QLabel(self)
        self.im_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.im1_width = 640
        self.im1_height = 480
        self.im_cam_label.resize(640, 420)
        self.im_cam_label.adjustSize()

        self.im_yolo_label = QLabel(self)
        self.im_yolo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.im1_width = 640
        self.im1_height = 480
        self.im_yolo_label.resize(640, 420)
        self.im_yolo_label.adjustSize()

        layout = QGridLayout()
        layout.addWidget(self.im_cam_label, 0, 0)
        layout.addWidget(self.im_yolo_label, 0, 3)
        layout.addWidget(self.start, 3, 0)
        layout.addWidget(self.stop, 3, 1)
        layout.addWidget(self.quit, 3, 2)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # image from camera
        # create the video capture thread
        self.camera_thread = VideoThread()
        # connect its signal to the update_image slot
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.camera_thread.start()

        self.robot_thread = RobotThread(self.camera_thread)
        # connect its signal to the update_image slot
        self.robot_thread.yolo_signal.connect(self.update_yolo_image)
        # start the thread
        self.robot_thread.start()

    def convert_cv_qt(self, cv_img):
        # convert from an opencv image to QPixmap
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.im1_width, self.im1_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def yolo_convert_cv_qt(self, cv_img):
        # convert from an opencv image to QPixmap
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.im1_width, self.im1_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # updates the image_label with a new opencv image
        img = self.convert_cv_qt(cv_img)
        self.im_cam_label.setPixmap(img)

    @pyqtSlot(np.ndarray)
    def update_yolo_image(self, cv_img):
        # updates the image_label with a new opencv image
        yolo_img = self.yolo_convert_cv_qt(cv_img)
        self.im_yolo_label.setPixmap(yolo_img)

    def start_clicked(self):
        self.start.setDisabled(True)
        self.stop.setDisabled(False)
        print("Start button!")
        self.robot_thread.is_work = True
        #self.robot_thread.sorting(self.camera_thread)

    def stop_clicked(self):
        self.start.setDisabled(False)
        self.stop.setDisabled(True)
        self.robot_thread.is_work = False
        print("Stop button!")

    def quit_clicked(self):
        self.camera_thread.stop()
        self.robot_thread.stop()
        self.close()
        print("Quit!")

    # def closeEvent(self, event):
    #     self.thread.stop()
    #     event.accept()

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
