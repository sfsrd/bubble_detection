import cv2
import numpy as np
import sys
import matplotlib as plt
from math import sqrt
import imutils

def lenBubbleSort(arr): 
    n = len(arr) 
    for i in range(n-1):  
        for j in range(0, n-i-1): 
            if len(arr[j]) < len(arr[j+1]) : 
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


img_original = cv2.imread('3.jpeg')

img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("gray.jpg", thresh1)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,7)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)


items = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(items) 

th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
th2 = cv2.drawContours(th2, contours, -1, (0,0,255), 2, cv2.LINE_AA)
cv2.imwrite("th2.jpg", th2)


#contours = sorted(contours, key = cv2.contourArea, reverse = True)
contours = lenBubbleSort(contours)
c = contours[0]
rbox = cv2.fitEllipse(c)
print('rbox', rbox)
x = int(rbox[0][0])
y = int(rbox[0][1])
h = int(rbox[1][0])
w = int(rbox[1][1])
print(x,y, h, w)

cv2.ellipse(img_original, rbox, (0, 255, 0), 2, cv2.LINE_AA)
image = cv2.circle(img_original, (x, y), radius=0, color=(255, 0, 0), thickness=3)

x,y,w,h  = cv2.boundingRect(c)
print(x,y, h, w)
#image = cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 0, 255), thickness=2) 
#image = cv2.circle(img_original, (int(0.5*w+x), int(0.5*h+y)), radius=0, color=(0, 0, 255), thickness=3)
image = cv2.drawContours(img_original, contours, 0, (153, 0, 153), 1, cv2.LINE_AA)

cv2.imshow("Result", img_original)

cv2.imwrite("Result.jpg", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
