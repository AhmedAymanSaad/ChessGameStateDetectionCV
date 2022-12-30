"""
This class is responsible for the common functions used in the project.
This includes all the image processing functions. 
"""

import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
import os
import math
import matplotlib.pyplot as plt
import skimage
import skimage.io as io

from matplotlib import cm
import cv2
from skimage import feature
from skimage import io, color, draw, transform, filters
from skimage.color import rgb2gray 
from skimage.filters import gaussian ,threshold_mean ,try_all_threshold ,threshold_otsu,threshold_triangle,threshold_minimum
from skimage.transform import rescale
from skimage.morphology import dilation,closing,opening,erosion, disk,rectangle
from sklearn.cluster import KMeans
from scipy.signal import convolve2d


from Common.ChessBoard import ChessBoard
from Common.Definitions import *


config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("Common\Config.ini")

def boolCast(value):
    """
    This function is responsible for casting a string to a boolean.
    :param value: The string to cast.
    :return: The boolean value.
    """
    if value == "True" or value == "true" or value == "1":
        return True
    elif value == "False" or value == "false" or value == "0":
        return False
    else:
        raise ValueError("The value is not a boolean.")

def tupleCast(value):
    """
    This function is responsible for casting a string to a tuple.
    :param value: The string to cast.
    :return: The tuple value.
    """
    #remove the brackets
    value = value[1:-1]
    return tuple(map(int, value.split(",")))