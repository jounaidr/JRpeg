import cv2
from matplotlib import pyplot as plt
import skimage.util
import numpy as np
from scipy.fftpack import dct, idct

def rgb_to_YCbCr(img):
    # height, width = img.shape[:2]  # Get height and width of image

    # for i in range(0, height):
    #     for j in range(0, width):
    #         b,g,r = img[i, j][:3]  # Get RGB values from image channels
    #
    #         img[i, j][0] = 0.299*r + 0.587*g + 0.114*b  # Y
    #         img[i, j][1] = (-0.1687*r - 0.3313*g + 0.5*b) + 128  # Cb
    #         img[i, j][2] = (0.5*r - 0.4187*g - 0.0813*b) + 128  # Cr

    b, g, r = cv2.split(img)  # Get RGB values from image channels

    img[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b  # Y
    img[:, :, 1] = (-0.1687 * r - 0.3313 * g + 0.5 * b) + 128  # Cb
    img[:, :, 2] = (0.5 * r - 0.4187 * g - 0.0813 * b) + 128  # Cr

    return img

def downsample_CbCr(img, n):
    cb_box = cv2.boxFilter(img[:, :, 1], ddepth=-1, ksize=(n, n))
    cr_box = cv2.boxFilter(img[:, :, 2], ddepth=-1, ksize=(n, n))

    cb_sub = cb_box[::n, ::n]
    cr_sub = cr_box[::n, ::n]

    return [np.float32(img[:,:,0]), np.float32(cb_sub), np.float32(cr_sub)]

def dtc_blocks(img_YCbCr):

    for n in range(3):
        #height, width = img_YCbCr[n].shape[:2]  # Get height and width of image
        #img_YCbCr[n] = img_YCbCr[n][: -(height % 8), : -(width % 8)]  # Ensure height and width are multiple of 8

        img_YCbCr[n] = skimage.util.view_as_blocks(img_YCbCr[n], block_shape=(8, 8))
        print(img_YCbCr[n].shape)

        block_height, block_width = img_YCbCr[n].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                block = img_YCbCr[n][i, j]

                block = np.float32(block)
                block = block - 128
                block = cv2.dct(block)
                img_YCbCr[n][i, j, :, :] = block

    print(img_YCbCr)


img = cv2.imread("./bpm-img/IC1.bmp")

img = rgb_to_YCbCr(img)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

img_YCbCr = downsample_CbCr(img, 2)

dtc_blocks(img_YCbCr)

# plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
# plt.subplot(1,3,1)
# plt.imshow(img_YCbCr[0],cmap="gray")
# plt.title('Luminance Y')
# plt.subplot(1,3,2)
# plt.imshow(img_YCbCr[1])
# plt.title('Subsampled Chrominance Cb')
# plt.subplot(1,3,3)
# plt.imshow(img_YCbCr[2])
# plt.title('Subsampled Chrominance Cr')
# plt.show()

# img = cv2.resize(img, (1440, 1080))
# cv2.imshow('Original', img)


