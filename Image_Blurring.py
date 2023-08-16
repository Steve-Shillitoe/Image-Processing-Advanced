import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image():
    img = cv2.imread('bricks.jpg').astype(np.float32)/255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_pic(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()
    
i = load_image()
show_pic(i)

#Apply Gamma Correction
gamma=0.25 #less than 1, so brighter image. greated than 1, darker image
gamma_img = np.power(i, gamma)
show_pic(gamma_img)

img = load_image()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='bricks', org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
show_pic(img)



