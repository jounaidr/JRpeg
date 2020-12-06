import cv2
import imutils as imutils
from matplotlib import pyplot as plt
import skimage.util
import numpy as np
from scipy.fftpack import dct, idct
from einops import rearrange
from itertools import groupby
from sys import getsizeof
import pickle

import util


def rgb_to_YCbCr(img):
    # height, width = img.shape[:2]  # Get height and width of image

    # for i in range(0, height):
    #     for j in range(0, width):
    #         b,g,r = img[i, j][:3]  # Get RGB values from image channels
    #
    #         img[i, j][0] = 0.299*r + 0.587*g + 0.114*b  # Y
    #         img[i, j][1] = (-0.1687*r - 0.3313*g + 0.5*b) + 128  # Cb
    #         img[i, j][2] = (0.5*r - 0.4187*g - 0.0813*b) + 128  # Cr

    r, g, b = cv2.split(img)  # Get RGB values from image channels

    img[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b  # Y
    img[:, :, 1] = (-0.1687 * r - 0.3313 * g + 0.5 * b) + 128  # Cb
    img[:, :, 2] = (0.5 * r - 0.4187 * g - 0.0813 * b) + 128  # Cr

    return np.uint8(img)


def downsample_CbCr(img, n):
    cb_box = cv2.boxFilter(img[:, :, 1], ddepth=-1, ksize=(n, n))
    cr_box = cv2.boxFilter(img[:, :, 2], ddepth=-1, ksize=(n, n))

    cb_sub = cb_box[::n, ::n]
    cr_sub = cr_box[::n, ::n]

    return [np.float32(img[:, :, 0]), np.float32(cb_sub), np.float32(cr_sub)]


def YCbCr_to_rgb(img_YCbCr):
    # original_width = img_YCbCr[0].shape[1]
    #
    # img_YCbCr[1] = imutils.resize(img_YCbCr[1], width=original_width)
    # img_YCbCr[2] = imutils.resize(img_YCbCr[2], width=original_width)
    #
    # output = np.asarray(img_YCbCr)
    # output = np.transpose(output, (1, 2, 0))
    #
    # xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    # rgb = output.astype(np.float)
    # rgb[:,:,[1,2]] -= 128
    # rgb = rgb.dot(xform.T)
    # np.putmask(rgb, rgb > 255, 255)
    # np.putmask(rgb, rgb < 0, 0)
    # return np.uint8(rgb)
    original_width = img_YCbCr[0].shape[1]

    img_YCbCr[1] = imutils.resize(img_YCbCr[1], width=original_width)
    img_YCbCr[2] = imutils.resize(img_YCbCr[2], width=original_width)

    output = np.asarray(img_YCbCr)
    output = np.transpose(output, (1, 2, 0))

    rgb = output.astype(np.float)

    y = output[:, :, 0]
    cb = output[:, :, 1] - 128
    cr = output[:, :, 2] - 128

    rgb[:, :, 0] = y + 1.402 * cr
    rgb[:, :, 1] = y - 0.34414 * cb - 0.71414 * cr
    rgb[:, :, 2] = y + 1.772 * cb

    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)

    return np.uint8(rgb)


def dtc_blocks(img_YCbCr):
    for n in range(3):
        # height, width = img_YCbCr[n].shape[:2]  # Get height and width of image
        # img_YCbCr[n] = img_YCbCr[n][: -(height % 8), : -(width % 8)]  # Ensure height and width are multiple of 8

        img_YCbCr[n] = skimage.util.view_as_blocks(img_YCbCr[n], block_shape=(8, 8))
        # print(img_YCbCr[n].shape)

        block_height, block_width = img_YCbCr[n].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                block = img_YCbCr[n][i, j]

                block = np.float32(block)
                block = block - 128
                block = cv2.dct(block)

                if n == 0:
                    block = np.trunc(block / util.Qlum)
                else:
                    block = np.trunc(block / util.Qchrom)

                img_YCbCr[n][i, j] = block

    print("poop")


def inverse_dtc_blocks(img_YCbCr):

    for n in range(3):

        block_height, block_width = img_YCbCr[n].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                block = img_YCbCr[n][i, j]

                if n == 0:
                    block = block * util.Qlum
                else:
                    block = block * util.Qchrom

                block = cv2.dct(block, flags=1)
                block = block + 128

                img_YCbCr[n][i, j] = block

        img_YCbCr[n] = rearrange(img_YCbCr[n], 'x y dx dy -> (x dx) (y dy)')
        np.trunc(img_YCbCr[n])

        print(img_YCbCr[n].shape)

    return img_YCbCr


def zigzag_block(block):
    height, width = block.shape[:2]

    block_string = [None] * 64

    for i in range(height):
        for j in range(width):
            block_string[util.zigzag_idx[i][j]] = block[i][j]

    return block_string


def encode_quantised_dct_img(img):
    encoded_list = [[], [], []]

    print(getsizeof(img))

    for n in range(3):
        block_height, block_width = img_YCbCr[n].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                block = img[n][i, j]
                #img[n][i, j] = zigzag_block(block)
                encoded_list[n] += zigzag_block(block)

        print(min(encoded_list[n]), " + ", max(encoded_list[n]))

        encoded_list[n] = list(map(int, encoded_list[n]))

        encoded_list[n] = [[len(list(group)), key] for key, group in groupby(encoded_list[n])]
        encoded_list[n].extend([block_height,block_width])

    #encoded_list = list(map(np.int8, encoded_list))

    pickle.dump(encoded_list, open("encoded.bin", "wb"))

    print(getsizeof(encoded_list))

def un_zigzag_block(block_string):
    block = np.zeros((8,8))

    for i in range(8):
        for j in range(8):
            block[i,j] = block_string[util.zigzag_idx[i][j]]

    return block

def decode_quantised_dct_img():
    compressed_img = pickle.load(open("encoded.bin", "rb"))
    encoded_list = [[], [], []]

    for i in range(3):
        block_width = compressed_img[i].pop()
        block_height = compressed_img[i].pop()

        out_blocks = np.zeros((block_height,block_width,8,8))
        for j in range(len(compressed_img[i])):
            num, value = compressed_img[i][j]
            num = int(num)
            encoded_list[i].extend(np.full(num, value))

        chunks = len(encoded_list[i]) / 64
        encoded_list[i] = np.array_split(encoded_list[i], chunks)

        pos = 0
        for x in range(0, block_height):
            for y in range(0, block_width):
                out_blocks[x,y] = un_zigzag_block(encoded_list[i][pos])
                pos = pos + 1

        encoded_list[i] = out_blocks
        print("yeet")

    return encoded_list



img = cv2.imread("./bpm-img/IC1.bmp")
original = cv2.imread("./bpm-img/IC1.bmp")

pickle.dump(original, open("original.bin", "wb"))
print(getsizeof(original))

img_YCbCr = rgb_to_YCbCr(img)


img_YCbCr = downsample_CbCr(img_YCbCr, 2)

dtc_blocks(img_YCbCr)

encode_quantised_dct_img(img_YCbCr)

# decode_quantised_dct_img()
#
# ya_boi_did_it = decode_quantised_dct_img()
#
# ya_boi_did_it = inverse_dtc_blocks(ya_boi_did_it)
#
# output_rgb = YCbCr_to_rgb(ya_boi_did_it)

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

# original = cv2.resize(original, (1440, 1080))
# cv2.imshow('Original', original)

# output_rgb = cv2.resize(output_rgb, (1440, 1080))
# cv2.imshow('compressed out', output_rgb)

cv2.waitKey(0)
