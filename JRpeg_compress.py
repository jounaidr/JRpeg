# @author jounaidr
# Source: https://github.com/jounaidr/JRpeg
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import pickle
from itertools import groupby
from sys import getsizeof

import cv2
import numpy as np
import skimage.util

import JRpeg_util


def rgb_to_ycbcr(img):
    r, g, b = cv2.split(img)  # Get RGB values from image channels

    # YCbCr conversion: https://www.mir.com/DMG/ycbcr.html
    img[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b  # Y
    img[:, :, 1] = (-0.1687 * r - 0.3313 * g + 0.5 * b) + 128  # Cb
    img[:, :, 2] = (0.5 * r - 0.4187 * g - 0.0813 * b) + 128  # Cr

    return np.uint8(img)  # Return image with Y,Cb,Cr in place of R,G,B channels, as an 8 bit unsigned integer


def down_sample_cbcr(YCbCr_img, sample_factor):
    # Averaging box filter on Cb and Cr channels, with mask size sample_factor x sample_factor
    cb_box = cv2.boxFilter(YCbCr_img[:, :, 1], ddepth=-1, ksize=(sample_factor, sample_factor))
    cr_box = cv2.boxFilter(YCbCr_img[:, :, 2], ddepth=-1, ksize=(sample_factor, sample_factor))

    # Down size Cb and Cr channels by sample_factor
    cb_sub = cb_box[::sample_factor, ::sample_factor]
    cr_sub = cr_box[::sample_factor, ::sample_factor]

    return [np.float32(YCbCr_img[:, :, 0]), np.float32(cb_sub), np.float32(cr_sub)]  # Return list containing Y, Cb and Cr components


def dtc_and_quantise_img(img, QL_rate, QC_rate):
    # For each channel (Y,Cb and Cr)
    for n in range(3):

        # Get height and width of image
        height, width = img[n].shape[:2]
        mod1 = (height % 8)
        mod2 = (width % 8)
        if ((height % 8) > 0) or ((width % 8) > 0):
            # Adjust height and width so they are a multiple of 8
            img[n] = img[n][:len(img[n]) -(height % 8), :len(img[n][0]) -(width % 8)]

        # Split component into array of 8x8 blocks
        img[n] = skimage.util.view_as_blocks(img[n], block_shape=(8, 8))
        # Get height and width of the blocks
        block_height, block_width = img[n].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                # For each block...
                block = img[n][i, j]

                # ...Convert values to float, and adjust so in range -128 to 128, then calculate the dct of the block
                block = cv2.dct((np.float32(block) - 128))

                if n == 0:
                    #  If Y channel divide by luminance_quantisation_matrix x luminance_quantisation_rate
                    block = np.trunc(block / (JRpeg_util.Qlum * QL_rate))
                else:
                    #  If Cb or Cr channels divide by chrominance_quantisation_matrix x chrominance_quantisation_rate
                    block = np.trunc(block / (JRpeg_util.Qchrom * QC_rate))
                # Set adjusted block in image
                img[n][i, j] = block

    return img #  Return split, dct and quantised image


def encode_and_save_quantised_dct_img(img_blocks, filename):
    #  Initialise new empty three channel list
    encoded_list = [[], [], []]

    # For each channel (Y,Cb and Cr)
    for n in range(3):
        # Get height and width of the blocks
        block_height, block_width = img_blocks[n].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                # For each block...
                block = img_blocks[n][i, j]
                # add the zigzag'd string to the current channel
                encoded_list[n] += zigzag_block(block)

        # Convert elements to int currently found to be optimal data type...
        # ...However possibly an 8 bit signed data type could be better (numpy int8 tested but worse than int...)
        encoded_list[n] = list(map(int, encoded_list[n]))
        # RLE style grouping of elements, will create tuples of (amount, value), for example [0,0,0,0] -> (4, 0)
        encoded_list[n] = [[len(list(group)), key] for key, group in groupby(encoded_list[n])]
        # Append list with meta data (block height, block width, QLrate and QCrate)
        encoded_list[n].extend([block_height,block_width])

    # Save list to binary file
    pickle.dump(encoded_list, open(filename, "wb"))

    return encoded_list  # Return the encoded image as a list


def zigzag_block(block):
    # Initialise an empty list of size 64
    block_string = [None] * 64

    for i in range(8):
        for j in range(8):
            # Set the (i,j) element in the block to the position in the block_string defined by (i,j) element from zigzag_idx
            block_string[JRpeg_util.zigzag_idx[i][j]] = block[i][j]

    return block_string  # Return the zigzag'd block as single list


# Default params:
#   -cbcr_downsize_rate=2, any higher will be noticeable, and greater than 5 will give diminishing reduction in file size
#   -QL_rate=1, standard JPEG luminance quantisation
def JRpeg_compress(input_filename, output_filename="encoded_img", cbcr_downsize_rate=2, QL_rate=16, QC_rate=1):
    # Read in original image as RGB three channel array
    original_img = cv2.imread(input_filename)
    print("Original image in memory size: ",getsizeof(original_img),"bytes")

    # Convert image to YCbCr and downsample Cb and Cr channels
    img_YCbCr = rgb_to_ycbcr(original_img)
    img_YCbCr_downsampled = down_sample_cbcr(img_YCbCr, cbcr_downsize_rate)
    print("YCbCr downsized image in memory size: ", getsizeof(img_YCbCr_downsampled), "bytes")

    # Convert YCbCr image into 8x8 blocks and calculate dct on each block, then quantise each block
    quantised_dct_img = dtc_and_quantise_img(img_YCbCr_downsampled, QL_rate, QC_rate)
    # Encode quantised dct YCbCr image with RLE grouping, and save to a binary file
    encoded_img = encode_and_save_quantised_dct_img(quantised_dct_img, (output_filename + ".bin"))

    # Print in memory size of encoded list in bytes
    print("Encoded image in memory size: ",getsizeof(encoded_img),"bytes")


JRpeg_compress("./bpm-img/IC1.bmp") #TODO: add qlrate and qcrate to encoder and pull out in decoder!!!!!!!!!!!!!!!!!!!!!!!!!!!
