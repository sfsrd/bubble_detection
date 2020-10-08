import cv2
import numpy as np

fpath = "img2.jpeg"
img = cv2.imread(fpath)

## Convert into grayscale and threshed it
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th, threshed = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

## Morph to denoise
threshed = cv2.dilate(threshed, None)
threshed = cv2.erode(threshed, None)

## Find the external contours
cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
cv2.drawContours(img, cnts, -1, (255, 0, 0), 2, cv2.LINE_AA)

## Fit ellipses
for cnt in cnts:
    if cnt.size < 10 or cv2.contourArea(cnt) < 100:
        continue

    rbox = cv2.fitEllipse(cnt)
    cv2.ellipse(img, rbox, (255, 100, 255), 2, cv2.LINE_AA)

## This it
cv2.imwrite("dst.jpg", img)

