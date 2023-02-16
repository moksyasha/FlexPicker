import cv2 as cv


def main():
    cam_port = 1
    cam = cv.VideoCapture(cam_port)

    while True:
        result, image = cam.read()
        if result:
            cv.imshow("test", image)
            key = cv.waitKey(1)
            if key%256 == 32:
                # SPACE pressed
                cv.destroyWindow("test")
                break
        else:
            print("No image detected. Please! try again")
            break

    cam.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()


