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
import os

import cv2
import numpy as np

Qlum=np.array([[16,11,10,16,24,40,51,61],
               [12,12,14,19,26,48,60,55],
               [14,13,16,24,40,57,69,56],
               [14,17,22,29,51,87,80,62],
               [18,22,37,56,68,109,103,77],
               [24,35,55,64,81,104,113,92],
               [49,64,78,87,103,121,120,101],
               [72,92,95,98,112,100,103,99]])

Qchrom=np.array([[17,18,24,47,99,99,99,99],
                 [18,21,26,66,99,99,99,99],
                 [24,26,56,99,99,99,99,99],
                 [47,66,99,99,99,99,99,99],
                 [99,99,99,99,99,99,99,99],
                 [99,99,99,99,99,99,99,99],
                 [99,99,99,99,99,99,99,99],
                 [99,99,99,99,99,99,99,99]])

zigzag_idx = np.array([[0,1,5,6,14,15,27,28],
                       [2,4,7,13,16,26,29,42],
                       [3,8,12,17,25,30,41,43],
                       [9,11,18,24,31,40,44,53],
                       [10,19,23,32,39,45,52,54],
                       [20,22,33,38,46,51,55,60],
                       [21,34,37,47,50,56,59,61],
                       [35,36,48,49,57,58,62,63]])


def zigzag_block(block):
    # Initialise an empty list of size 64
    block_string = [None] * 64

    for i in range(8):
        for j in range(8):
            # Set the (i,j) element in the block to the position in the block_string defined by (i,j) element from zigzag_idx
            block_string[zigzag_idx[i][j]] = block[i][j]

    return block_string  # Return the zigzag'd block as single list


def un_zigzag_block(block_list):
    # Generate an empty 8x8 block
    block = np.zeros((8,8))

    # For each element in the block...
    for i in range(8):
        for j in range(8):
            # ...set the (i,j) element of the block
            # ...To the element from the block_list with the corresponding index...
            # ...define by (i,j) zigzag_idx
            block[i,j] = block_list[zigzag_idx[i][j]]

    return block  # Return the formatted 8x8 block

def get_img_disk_size(filename):
    # Get size of file on disk
    file = os.stat(filename)
    return file.st_size

def meanSquareError(original_filename, compressed_filename):
    imgA = cv2.imread(original_filename)
    imgB = cv2.imread(compressed_filename)

    mse = np.square(np.subtract(imgA, imgB)).mean()

    return mse