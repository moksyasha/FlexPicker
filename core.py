import image_processing as img_proc

import cv2 as cv
import numpy as np
import socket
import pyrealsense2 as rs
import pandas as pd
import time

from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMainWindow, QPushButton
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from threading import Thread
from threading import Event

import pyzbar.pyzbar as pyzbar
from matplotlib import pyplot as plt

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_PT = ROOT / "nn.pt"
stop_thread = False


def get_command(cam_point, trans):
    cp_new = np.append(cam_point, [1])
    rp2 = np.dot(trans, cp_new.T)
    rp2[0] -= 60
    rp2[1] -= 15
    rp2[2] += 17

    stri = np.array2string(rp2, formatter={'float_kind':lambda x: "%.2f" % x})
    cmd = "MJ_ARC " + stri[1:-1]
    return cmd, rp2


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Delta-robot sorting")
        self.display_width = 1300
        self.display_height = 750
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)
        # create a text label
        self.textLabel = QLabel('Affine transform')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)


class MyThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None):
	    super().__init__(group=group, target=target, 
			              name=name)
	    self.args = args
	    self.kwargs = kwargs
	    return

    def run(self):
        print("Camera thread started!")
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs

class Camera():
    def __init__(self):
        self.database = pd.read_csv("database.csv")
        self.cat = 0
        self.camera = cv.VideoCapture(3)
        self.camera.set(cv.CAP_PROP_FPS, 30)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        self.stop_event = Event()
        self.thread_camera = Thread(target=self.camera_reading, args=(self.camera, self.stop_event))

    def start_thread(self):
        self.thread_camera.start()

    def stop_thread(self):
        self.stop_event.set()
        self.thread_camera.join()
        self.camera.release()

    def decodeBar(self, image):
        barcode_detector = cv.barcode_BarcodeDetector()
        # 'retval' is boolean mentioning whether barcode has been detected or not
        retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(image)

        #print(type(decoded_info[0]))
        # copy of original image

        # proceed further only if at least one barcode is detected:
        if retval:
            points = points.astype(np.int32)
            for i, point in enumerate(points):
                if decoded_info[i] == "":
                    continue

                image = cv.drawContours(image, [point], 0, (0, 255, 0), 2)

                # uncomment the following to print decoded information
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

    def camera_reading(self, camera, stop_event):
        while True:
            ret, frame = camera.read()
            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            im = self.decodeBar(frame)
            cv.imshow("camera", im)
            cv.waitKey(10)
            #out.write(im)
            if self.stop_event.is_set():
                print("stop reading!")
                break
    
    def get_cat(self):
        return self.cat

class Robot():
    def __init__(self):
        self.matrix = np.loadtxt("matrix_rigid.txt")
        
        # #create socket, connect to robot controller and send data
        self.sock = socket.socket()
        self.sock.connect(("192.168.125.1", 1488))

        # # Load model
        device = select_device("")
        self.model = DetectMultiBackend(PATH_PT, device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((1280, 736), s=stride)  # check image size

        # # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        #rs.config.enable_device_from_file(config, "1.bag") ## DEL
        #config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        #config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 4) # High density preset

        align_to = rs.stream.depth
        self.align = rs.align(align_to)
        self.fltr = rs.hole_filling_filter(2)

    def get_str_command(self, cam_point):
        stri = np.array2string(cam_point, formatter={'float_kind':lambda x: "%.2f" % x})
        cmd = "MJ_ARC " + stri[1:-1]
        return cmd

    def cmd_to_robot(self, sock, cmd):
        sock.send(cmd.encode('ASCII'))
        data = sock.recv(1024)
        print(data.decode('ASCII') + " :" + cmd)

    def main_sorting(self, camera):
        is_work = 1
        while(is_work):
            try:
                cur_category = 0
                self.cmd_to_robot(self.sock, "MJ 0 0 250")
                #cmd_to_robot(sock, "VALVE_OPEN ")

                for x in range(5):
                    frames = self.pipeline.wait_for_frames()
                
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_frame_fltr = self.fltr.process(depth_frame)

                # convert camera image to robot space
                color_frame_affine, rbt_verts = img_proc.get_img_affine(color_frame, depth_frame_fltr, self.matrix)

                pad = np.ones((16, 1280, 3), dtype=np.uint8)
                color_frame_affine = np.append(color_frame_affine, pad, axis=0)

                x, y, angle = img_proc.get_center(color_frame_affine, self.model, 1)

                if x == 0:
                    print("Didnt find object!")
                    continue

                # plt.figure(figsize=(12, 4))
                # plt.subplot(131)
                # plt.imshow(color_frame_affine)
                # plt.subplot(132)
                # plt.imshow(rbt_verts)
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
                new_t = np.array([-455, -459, 0])
                trans_rbt_pnt = np.array([x, y, depth]) + new_t

                cmd_box = self.get_str_command(trans_rbt_pnt)

                # # # rotate manipulator before taking
                self.cmd_to_robot(self.sock, "ROT " + str(angle))

                # take a box
                self.cmd_to_robot(self.sock, "PUMP_START ")
                self.cmd_to_robot(self.sock, cmd_box)
                time.sleep(1.5)

                # pick box up
                trans_rbt_pnt[2] = 250
                stri = np.array2string(trans_rbt_pnt, formatter={'float_kind':lambda x: "%.2f" % x})
                cmd = "MJ " + stri[1:-1]
                self.cmd_to_robot(self.sock, cmd)

                # rotate manipulator after taking for alignment
                self.cmd_to_robot(self.sock, "ROT " + str(-angle))

                # moving to camera for detecting !! изменить координаты камеры
                self.cmd_to_robot(self.sock, "MJ -302 -478 252")
                #cmd_to_robot(sock, "MJ -165 -228 33")

                # detecting qr
                for i in range(1):
                    time.sleep(2.5)
                    cur_category = camera.get_cat()
                    if cur_category!=0:
                        break
                    self.cmd_to_robot(self.sock, "ROT 90")
                    time.sleep(2.5)
                    cur_category = camera.get_cat()
                    if cur_category!=0:
                        break
                    self.cmd_to_robot(self.sock, "ROT 90")
                    time.sleep(2.5)
                    cur_category = camera.get_cat()
                    if cur_category!=0:
                        break
                    self.cmd_to_robot(self.sock, "ROT 90")
                    time.sleep(2.5)
                    cur_category = camera.get_cat()

                if cur_category == 1:
                    self.cmd_to_robot(self.sock, "MJ -302 -200 60") # найдено в бд
                else:
                    print("not")

                # # TODO category
                #self.cmd_to_robot(self.sock, "MJ -302 -200 60")
                self.cmd_to_robot(self.sock, "8VALVE_1 ") # open value
                self.cmd_to_robot(self.sock, "PUMP_STOP ")
                self.cmd_to_robot(self.sock, "8VALVE_0 ") # close default
                self.cmd_to_robot(self.sock, "ROT_BASE ")

                is_work -= 1

            except Exception as e:
                print("Something went wrong")
                print(e)
                is_work = 0

        # stop_event.set()
        # thread_camera.join()
        self.cmd_to_robot(self.sock, "MJ 0 0 250")
        self.cmd_to_robot(self.sock, "PUMP_STOP ")
        self.cmd_to_robot(self.sock, "8VALVE_0 ")
        return



def main():
    robot = Robot()
    camera = Camera()
    camera.start_thread()
    robot.main_sorting(camera)
    camera.stop_thread()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
    # app = QApplication([])
    # window = App()
    # window.show()
    # sys.exit(app.exec())