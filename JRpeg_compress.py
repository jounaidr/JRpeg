# Author: jounaidr
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

import logging
logging.basicConfig(
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def rgb_to_ycbcr(img):
    r, g, b = cv2.split(img)  # Get RGB values from image channels

    # YCbCr conversion: https://www.mir.com/DMG/ycbcr.html
    img[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b  # Y
    img[:, :, 1] = (-0.1687 * r - 0.3313 * g + 0.5 * b) + 128  # Cb
    img[:, :, 2] = (0.5 * r - 0.4187 * g - 0.0813 * b) + 128  # Cr

    return np.uint8(img)  # Return image with Y,Cb,Cr in place of R,G,B channels, as an 8 bit unsigned integer


def down_sample_cbcr(YCbCr, sample_factor):
    if sample_factor > 1:
        # Averaging box filter on Cb and Cr channels, with mask size sample_factor x sample_factor
        cb_box = cv2.boxFilter(YCbCr[:, :, 1], ddepth=-1, ksize=(sample_factor, sample_factor))
        cr_box = cv2.boxFilter(YCbCr[:, :, 2], ddepth=-1, ksize=(sample_factor, sample_factor))

        # Down size Cb and Cr channels by sample_factor
        cb_down = cb_box[::sample_factor, ::sample_factor]
        cr_down = cr_box[::sample_factor, ::sample_factor]

        return [np.float32(YCbCr[:, :, 0]), np.float32(cb_down), np.float32(cr_down)]  # Return list containing Y with downsampled Cb and Cr components as float

    return [np.float32(YCbCr[:, :, 0]), np.float32(YCbCr[:, :, 1]), np.float32(YCbCr[:, :, 2])]  # If not downsampling selected just return Y, Cb and Cr in list format


def dtc_and_quantise_img(img, QL_rate, QC_rate):
    # For each channel (Y,Cb and Cr)
    for ch in range(3):

        # Get height and width of image
        height, width = img[ch].shape[:2]
        if ((height % 8) > 0) or ((width % 8) > 0):
            # Adjust height and width so they are a multiple of 8
            img[ch] = img[ch][:len(img[ch]) -(height % 8), :len(img[ch][0]) -(width % 8)]

        # Split component into array of 8x8 blocks
        img[ch] = skimage.util.view_as_blocks(img[ch], block_shape=(8, 8))
        # Get height and width of the blocks
        block_height, block_width = img[ch].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                # For each block...
                block = img[ch][i, j]

                # ...Convert values to float, and adjust so in range -128 to 128, then calculate the dct of the block...
                # ...More info: https://docs.opencv.org/2.4.3/modules/core/doc/operations_on_arrays.html?highlight=dct#cv.DCT
                block = cv2.dct((np.float32(block) - 128))

                if ch == 0:
                    #  If Y channel divide by luminance_quantisation_matrix x luminance_quantisation_rate (if not 0)
                    if QL_rate != 0:
                        block = np.trunc(block / (JRpeg_util.Qlum * QL_rate))
                else:
                    #  If Cb or Cr channels divide by chrominance_quantisation_matrix x chrominance_quantisation_rate (if not 0)
                    if QC_rate != 0:
                        block = np.trunc(block / (JRpeg_util.Qchrom * QC_rate))
                # Set adjusted block in image
                img[ch][i, j] = block

    return img #  Return split, dct and quantised image


def encode_and_save_quantised_dct_img(img_blocks, QL_rate, QC_rate, filename):
    # Initialise new empty three channel list
    encoded_list = [[], [], []]

    # For each channel (Y,Cb and Cr)
    for ch in range(3):
        # Get height and width of the blocks
        block_height, block_width = img_blocks[ch].shape[:2]

        for i in range(0, block_height):
            for j in range(0, block_width):
                # For each block...
                block = img_blocks[ch][i, j]
                # add the zigzag'd string to the current channel
                encoded_list[ch] += JRpeg_util.zigzag_block(block)

        # Convert elements to int currently found to be optimal data type...
        # ...However possibly an 8 bit signed data type could be better (numpy int8 tested but worse than int...)
        encoded_list[ch] = list(map(int, encoded_list[ch]))
        # RLE style grouping of elements, will create tuples of (amount, value), for example [0,0,0,0] -> (4, 0)
        encoded_list[ch] = [[len(list(group)), key] for key, group in groupby(encoded_list[ch])]

        # TODO: following can be optimised...
        for x in range(len(encoded_list[ch])):
            # Unpack the tuple element into separate variables for its value and number of occurrence
            num, value = encoded_list[ch][x]
            # If the occurrence of a value is just 1, just store its value instead of a tuple (half the size)
            if num == 1:
                encoded_list[ch][x] = value

        # Append channel with meta data for block height and block width, QLrate and QCrate
        encoded_list[ch].extend([ block_height, block_width])
    # Append luminance channel with meta data for block QLrate and QCrate
    encoded_list[0].extend([QL_rate, QC_rate])

    # Save list to binary file
    pickle.dump(encoded_list, open(filename, "wb"))

    return encoded_list  # Return the encoded image as a list


# Default params:
#   -cbcr_downsize_rate=2, any higher will be noticeable, and greater than 5 will give diminishing reduction in file size
#   -QL_rate=1, standard JPEG luminance quantisation todo: UPDATE THIS COMMENT WHEN SHITS FIXED LMAO
def JRpeg_compress(input_filename, output_filename="JRpeg_encoded_img.bin", cbcr_downsize_rate=2, QL_rate=1, QC_rate=1):
    # Read in original image as RGB three channel array and save a resized copy for display later
    logging.info("Loading original image file: {} ...".format(input_filename))
    original_img = cv2.imread(input_filename)
    logging.info("Original image in memory size: {} bytes!".format(getsizeof(original_img)))
    resized_original_img = cv2.resize(original_img, (1440, 1080))
    logging.info("... image loaded successfully!")

    # Convert image to YCbCr and downsample Cb and Cr channels
    logging.info("Converting RGB image to YCbCr image, with Cb and Cr downsampled by a factor of: {} ...".format(cbcr_downsize_rate))
    YCbCr = rgb_to_ycbcr(original_img)
    YCbCr_downsampled = down_sample_cbcr(YCbCr, cbcr_downsize_rate)
    logging.info("... YCbCr conversion and downsampling successful!")

    # Convert YCbCr image into 8x8 blocks and calculate dct on each block, then quantise each block
    logging.info("Attempting to DCT and quantise YCbCr with QLuminance_rate: {}, and QChrominance_rate {} ...".format(QL_rate,QC_rate))
    quantised_dct_img = dtc_and_quantise_img(YCbCr_downsampled, QL_rate, QC_rate)
    logging.info("... YCbCr DCT and quantisation successful!")

    # Encode quantised dct YCbCr image with RLE grouping, and save to a binary file
    logging.info("Attempting to encode and save JRpeg image as: {} ...".format(output_filename))
    encoded_img = encode_and_save_quantised_dct_img(quantised_dct_img, QL_rate, QC_rate, output_filename)
    logging.info("Encoded image in memory size: {} bytes!".format(getsizeof(encoded_img)))
    logging.info("... JRpeg image saved successfully!")

    # Display original input image
    cv2.imshow('Original Image: ' + input_filename, resized_original_img)
    cv2.waitKey(0)

    logging.info("############################################################################")
    logging.info("############################################################################")
    logging.info("## THANK YOUR FOR USING JRpeg, brought to you by: github.com/jounaidr !!! ##")
    logging.info("############################################################################")
    logging.info("############################################################################")

    return [getsizeof(original_img), getsizeof(encoded_img), JRpeg_util.get_img_disk_size(input_filename), JRpeg_util.get_img_disk_size(output_filename)]

# TODO: Add debug logging to methods