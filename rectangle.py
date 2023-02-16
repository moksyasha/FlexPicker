import cv2
import numpy as np


def left_click_detect(event, x, y, flags, points):
    if (event == cv2.EVENT_LBUTTONDOWN):
        print(f"\tClick on {x}, {y}")
        points.append([x,y])
        print(points)

def main():
    cap = cv2.VideoCapture(0)
    polygon = []
    points = []

    output = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*'MPEG'),
        30, (1080, 1920))

    while (True):
        ret, frame = cap.read()
        if (ret):
            fill_color = [255, 255, 255]  # any BGR color value to fill with
            mask_value = 255
            stencil = np.zeros(frame.shape[:-1]).astype(frame.dtype)
            frame = cv2.polylines(frame, polygon, True, (255, 255, 255), thickness=5)
            cv2.fillPoly(stencil, polygon, mask_value)
            result = frame
            if polygon != []:
                sel = stencil != mask_value
                frame[sel] = fill_color
            output.write(result)
            cv2.imshow('Frame', result)
            key = cv2.waitKey(25)
            if (key == ord('q')):
                break
            elif (key == ord('p')):
                polygon = [np.int32(points)]
                points = []

            cv2.setMouseCallback('Frame', left_click_detect, points)

    cv2.destroyAllWindows()
    output.release()
    cap.release()


if __name__ == "__main__":
    main()