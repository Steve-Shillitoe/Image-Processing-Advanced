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

def write_text_on_image():
    img = load_image()
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, text='bricks', org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
    return img
    

i = load_image()
show_pic(i)

#Apply Gamma Correction
gamma=0.25 #less than 1, so brighter image. greated than 1, darker image
gamma_img = np.power(i, gamma)
show_pic(gamma_img)

show_pic(write_text_on_image())

kernal = np.ones(shape=(5,5), dtype=np.float32)/25
img = write_text_on_image()
dst = cv2.filter2D(img, -1, kernal)
show_pic(dst)

img = write_text_on_image()
blurred = cv2.blur(img, ksize=(5,5))
show_pic(blurred)

img = write_text_on_image()
blurred = cv2.GaussianBlur(img,(5,5),10)
show_pic(blurred)

img = write_text_on_image()
median_blur = cv2.medianBlur(img,5)
show_pic(median_blur)

img = cv2.imread('sammy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_noise = cv2.imread('sammy_noise.jpg')
show_pic(img_noise)
median_blur = cv2.medianBlur(img_noise,5)
show_pic(median_blur)



