# py -m pip install opencv-contrib-python
import cv2
import pyzbar.pyzbar as pyzbar
from matplotlib import pyplot as plt
import numpy as np

def first(img):
    barcode_detector = cv2.barcode_BarcodeDetector()

    # 'retval' is boolean mentioning whether barcode has been detected or not
    retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(img)

    # copy of original image
    img2 = img.copy()

    # proceed further only if at least one barcode is detected:
    if retval:
        points = points.astype(np.int)
        for i, point in enumerate(points):

            img2 = cv2.drawContours(img2,[point],0,(0, 255, 0), 2)

            # uncomment the following to print decoded information
            x1, y1 = point[1]
            y1 = y1 - 10
            cv2.putText(img2, decoded_info[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, 2)
    return img2


def sec(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)

        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        # check barcode in database
        if barcodeData in database["code"].values:
            category = 1
            print("found in db")
        else:
            category = 0
            print("not found in db")
        
    return image


def main():
    img = cv2.imread('3.jpg')
    res1 = first(img)
    res2 = sec(img)
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(res1)
    plt.subplot(122)
    plt.imshow(res2)
    plt.show()


if __name__ == '__main__':
    main()
