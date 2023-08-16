import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#Blending images of the same size
#To use cv2.addWeighted, images must be the same size
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

blended = cv2.addWeighted(src1=img1, alpha=0.75, src2=img2, beta=0.25, gamma=0)
plt.imshow(blended)
plt.show()

#Overlay small image on top of a large image (no blending)
#Numpy reassignment
img1 = cv2.imread('dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
small_image = cv2.resize(img2, (600, 600))
large_image = img1

x_offset = 0
y_offset = 0
x_end = x_offset + small_image.shape[1]
y_end = y_offset + small_image.shape[0]

large_image[y_offset:y_end, x_offset:x_end] = small_image
plt.imshow(large_image)
plt.show()

#Blend images of different sizes (blending)
img1 = cv2.imread('dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
small_image = cv2.resize(img2, (600, 600))
large_image = img1
x_offset = img1.shape[1] - small_image.shape[1]
y_offset = img1.shape[0] - small_image.shape[0]

rows, cols, channels = small_image.shape

roi = img1[y_offset:img1.shape[0], x_offset:img1.shape[1]]
#plt.imshow(roi)
#plt.show()

#Create mask
small_image_gray = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
#plt.imshow(small_image_gray, cmap='gray')
#plt.show()

#create inverse of mask
inv_small_image_mask = cv2.bitwise_not(small_image_gray)
# Ensure the small_image_mask has the same shape as white_background
inv_small_image_mask = np.expand_dims(inv_small_image_mask, axis=2)
inv_small_image_mask = inv_small_image_mask.repeat(3, axis=2)
white_background = np.full(small_image.shape, 255, dtype=np.uint8)
#plt.imshow(inv_small_image_mask, cmap='gray')
#plt.show()
#print(inv_small_image_mask.shape)

#colour channel has been removed, add it back in
white_background = np.full(small_image.shape, 255, dtype=np.uint8)
background = np.bitwise_or(white_background, white_background, ~inv_small_image_mask)
foreground = np.bitwise_or(small_image, small_image, inv_small_image_mask)

final_roi = np.bitwise_or(roi, foreground)
#plt.imshow(final_roi)
#plt.show()

large_image[y_offset:y_offset+small_image.shape[0], x_offset:x_offset+small_image.shape[1]] = final_roi
plt.imshow(large_image)
plt.show()


