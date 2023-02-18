#import cv2 as cv
import socket

def main():
    # create socket, connect to RobotStudio server and send data
    sock = socket.socket()
    sock.connect(("127.0.0.1", 1488))
    
    while True:
        cmd = input()
        sock.send(cmd.encode('ASCII'))

        if cmd == "exit":
            break
        
        data = sock.recv(1024)
        print(data.decode('ASCII'))

    # cam_port = 1
    # cam = cv.VideoCapture(cam_port)

    # while True:
    #     result, image = cam.read()
    #     if result:
    #         cv.imshow("test", image)
    #         key = cv.waitKey(1)
    #         if key%256 == 32:
    #             # SPACE pressed
    #             cv.destroyWindow("test")
    #             break
    #     else:
    #         print("No image detected. Please! try again")
    #         break

    # cam.release()
    # cv.destroyAllWindows()

if __name__ == "__main__":
    main()


