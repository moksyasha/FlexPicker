import cv2
import numpy as np



def get_center(image):
    # Применим небольшое размытие для устранения шумов
    image = cv2.GaussianBlur(image, (7, 7), 0)
    # Переведём в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Сегментируем красный цвет
    saturation = hsv[...,1]
    saturation[(hsv[..., 0] > 15) & (hsv[..., 0] < 165)] = 0
    _, image1 = cv2.threshold(saturation, 92, 255, cv2.THRESH_BINARY)
    mask = image1
    # Найдем наибольшую связную область
    contours = cv2.findContours(image1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    # Оценим ее центр
    b_circle = cv2.minEnclosingCircle(contour)
    b = tuple(int(item) for item in b_circle[0])
    return b, mask

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")
cv2.namedWindow("test2")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    center, mask = get_center(frame)
    print(center)
    cv2.circle(frame, center, 1, (0, 0, 0), thickness=1, lineType=8, shift=0)
    cv2.imshow("test", frame)
    #cv2.circle(mask, center, 1, (255, 0, 0), thickness=1, lineType=8, shift=0)
    cv2.imshow("test2", mask)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(center)
        cv2.imwrite(img_name, frame)
        # print("{} written!".format(center))
        img_name2 = "opencv_mask_{}.png".format(center)
        cv2.imwrite(img_name2, mask)
        print("{} written!".format(center))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()