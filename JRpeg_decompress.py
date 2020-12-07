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

import JRpeg_util

def load_and_decode_quantised_dct_img(filename):
    # Load JRpeg binary file
    compressed_img = pickle.load(open(filename, "rb"))
    # Initialise new empty three channel list
    encoded_list = [[], [], []]

    #
    block_width = compressed_img[0].pop()
    block_height = compressed_img[0].pop()

    for i in range(3):
        # For each channel (Y,


        QC_rate = compressed_img[i].pop()
        QL_rate = compressed_img[i].pop()

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

    return [encoded_list, QL_rate, QC_rate]

def un_zigzag_block(block_string):
    block = np.zeros((8,8))

    for i in range(8):
        for j in range(8):
            block[i,j] = block_string[JRpeg_util.zigzag_idx[i][j]]

    return block