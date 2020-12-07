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

import cv2
import numpy as np
from einops import rearrange

import JRpeg_util

import logging
logging.basicConfig(
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def load_and_decode_quantised_dct_img(filename):
    # Load JRpeg binary file
    compressed_img = pickle.load(open(filename, "rb"))

    # Initialise new empty three channel list
    decoded_list = [[], [], []]
    # Retrieve meta data for luminance and chrominance quantify rates
    QC_rate = compressed_img[0].pop()
    QL_rate = compressed_img[0].pop()

    # For each channel (Y, Cb, Cr)...
    for i in range(3):
        # ...Retrieve meta data for blocks height and width
        block_width = compressed_img[i].pop()
        block_height = compressed_img[i].pop()
        # Generate an empty array of 8x8 blocks, of size block_height x block_width
        out_blocks = np.zeros((block_height,block_width,8,8))

        # For each element in the current channel...
        for j in range(len(compressed_img[i])):
            # Unpack the tuple element into separate variables for its value and number of occurrence
            num, value = compressed_img[i][j]
            # Fill the corresponding channel of the output list with the raw values...
            # ...For example: (5,0), (2,50), > [0,0,0,0,0,50,50]
            decoded_list[i].extend(np.full(int(num), value))
        # Split the current channel into list chunks of size 64 for each block
        chunks = len(decoded_list[i]) / 64
        decoded_list[i] = np.array_split(decoded_list[i], chunks)

        pos = 0
        # For each block
        for x in range(0, block_height):
            for y in range(0, block_width):
                # Reformat the current block from a single 64 element list (defined by pos)...
                # ...into the correct block format ("un-zigzag") and add the block to the out_blocks array (for the current channel)
                out_blocks[x,y] = JRpeg_util.un_zigzag_block(decoded_list[i][pos])
                pos = pos + 1
        # Replace the current channel list with the reformatted 8x8 blocks array
        decoded_list[i] = out_blocks

    return [decoded_list, QL_rate, QC_rate]  # Return the decoded list, QL_rate, QC_rate


def inverse_dct_blocks(decoded_list):
    # Get quantised DCT YCbCr image and quantisation metadata
    YCbCr = decoded_list[0]
    QL_rate = decoded_list[1]
    QC_rate = decoded_list[2]

    # For each channel (Y, Cb, Cr)...
    for ch in range(3):
        # Get the blocks height and width
        block_height, block_width = YCbCr[ch].shape[:2]
        # For each block
        for i in range(0, block_height):
            for j in range(0, block_width):
                # Store the current block temporarily
                block = YCbCr[ch][i, j]

                if ch == 0:
                    #  If Y channel multiply by luminance_quantisation_matrix x luminance_quantisation_rate
                    block = block * (JRpeg_util.Qlum * QL_rate)
                else:
                    #  If Cb or Cr channels multiply by chrominance_quantisation_matrix x chrominance_quantisation_rate
                    block = block * (JRpeg_util.Qchrom * QC_rate)

                # Inverse DCT the block using cv2.dct with DCT_INVERSE flag...
                # ...More info: https://docs.opencv.org/2.4.3/modules/core/doc/operations_on_arrays.html?highlight=dct#cv.DCT
                block = cv2.dct(block, flags=1) + 128
                # Replace the adjusted block back into the block array at its corresponding position
                YCbCr[ch][i, j] = block
        # Combine the 8x8 blocks into a single 2d array (for each channel), and truncate the values
        YCbCr[ch] = rearrange(YCbCr[ch], 'x y dx dy -> (x dx) (y dy)')
        np.trunc(YCbCr[ch])

    return YCbCr # The inverse dct/quantised YCbCr image


def YCbCr_to_rgb(YCbCr):
    # Resize Cb and Cr components to original size
    original_height, original_width = YCbCr[0].shape[:2]
    YCbCr[1] = cv2.resize(YCbCr[1], (original_width, original_height))
    YCbCr[2] = cv2.resize(YCbCr[2], (original_width, original_height))
    # Convert YCbCr to numpy array and transpose to the cv2 format: (x, y, channel)
    transpose_YCbCr = np.transpose(np.asarray(YCbCr), (1, 2, 0))
    # Store YCbCr as a float
    rgb_out = transpose_YCbCr.astype(np.float)

    # Get Y, Cb and Cr components and adjust Cb and Cr
    y = transpose_YCbCr[:, :, 0]
    cb = transpose_YCbCr[:, :, 1] - 128
    cr = transpose_YCbCr[:, :, 2] - 128

    # Convert YCbCr to RGB, more info: https://www.mir.com/DMG/ycbcr.html
    rgb_out[:, :, 0] = y + 1.402 * cr
    rgb_out[:, :, 1] = y - 0.34414 * cb - 0.71414 * cr
    rgb_out[:, :, 2] = y + 1.772 * cb
    # Ensure pixels are in 0 to 255 range
    np.putmask(rgb_out, rgb_out > 255, 255)
    np.putmask(rgb_out, rgb_out < 0, 0)

    return np.uint8(rgb_out)  # Return RGB image as unit8 (8 bit unsigned integer)


def JRpeg_decompress(input_filename="encoded_img"):
    # Load the JRpeg file into a YCbCr list, extract the metadata, and format each channel into 8x8 blocks
    logging.info("Loading and decoding JRpeg file: {}.bin ...".format(input_filename))
    decoded_list_with_metadata = load_and_decode_quantised_dct_img(input_filename + ".bin")
    logging.info("... YCbCr loaded and decoded successfully!")

    # Inverse dct/quantise the YCbCr image
    logging.info("Attempting to inverse DCT and un-quantise YCbCr with QLuminance_rate: {}, and QChrominance_rate {} ...".format(decoded_list_with_metadata[1],decoded_list_with_metadata[2]))
    YCbCr = inverse_dct_blocks(decoded_list_with_metadata)
    logging.info("... YCbCr inverse DCT and un-quantisation successful!")

    # Resize the Cb and Cr channels if downsized, and convert the YCbCr image to RGB
    logging.info("Attempting to resize Cb and Cr channels, and convert YCbCr to RGB ...")
    rgb_output = YCbCr_to_rgb(YCbCr)
    logging.info("... Conversion to RGB successful!")

    # Display decompressed JRpeg image and save a BMP version it
    logging.info("Attempting to display decompressed JRpeg image and save BMP copy ...")
    cv2.imwrite(input_filename + "_bmp_copy.bmp", rgb_output)

    rgb_output = cv2.resize(rgb_output, (1440, 1080))
    cv2.imshow('Decompressed JRpeg image: ' + input_filename + ".bin", rgb_output)
    cv2.waitKey(0)
    logging.info("... BMP copy saved as: {}".format(input_filename + "_bmp_copy.bmp"))

    logging.info("############################################################################")
    logging.info("############################################################################")
    logging.info("## THANK YOUR FOR USING JRpeg, brought to you by: github.com/jounaidr !!! ##")
    logging.info("############################################################################")
    logging.info("############################################################################")



# JRpeg_decompress()
# TODO: Add debug logging to methods