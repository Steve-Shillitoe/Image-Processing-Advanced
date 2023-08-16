import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('rainbow.jpg',0)
#plt.imshow(img, cmap='gray')
#plt.show()

ret, thres1 = cv2.threshold(img,127,255, cv2.THRESH_TOZERO)
#plt.imshow(thres1, cmap='gray')
#plt.show()

img = cv2.imread('crossword.jpg',0)
#plt.imshow(img, cmap='gray')
#plt.show()

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()
 
ret, thres2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#show_pic(thres2)

thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,8)
show_pic(thres_adapt)

blended = cv2.addWeighted(src1=thres2, alpha=0.6, src2=thres_adapt, beta=0.4, gamma=0)
show_pic(blended)