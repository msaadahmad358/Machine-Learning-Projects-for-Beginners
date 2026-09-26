#1 - import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)

#2 - read image and gray
image = cv2.imread('coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('outputs/1_gray.jpg', gray)

#3 - blur
blur = cv2.GaussianBlur(gray, (11 , 11), 0)
cv2.imwrite('outputs/2_blur.jpg', blur)

#4 - detect edges
canny=cv2.Canny(blur, 30, 150, 3)
cv2.imwrite('outputs/3_canny.jpg',canny)

#5 - connect edges
dilated = cv2. dilate(canny, (1, 1), iterations=0)
cv2.imwrite('outputs/4_dilated.jpg', dilated)

#6 - calculate contour in image
(cnt, hieraarchy) = cv2. findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('outputs/5_rgb.jpg',rgb)

#7 - result
print('objects in image: ', len(cnt))