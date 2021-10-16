"""
Change the parameters of the program here.
"""
import cv2 as cv
import numpy as np

# define max image size, if images are bigger, image is rescaled so that the width and height are <= MAX_IMAGE_SIZE.
# A larger MAX_IMAGE_SIZE is more accurate but slower and requires more storage.
MAX_IMAGE_SIZE = 1080

# branching factor of the kmeans tree
K = 10
assert K > 1
# depth of the kmeans tree
L = 6
assert L > 0
# total leaf nodes = K^L


# reduces required disk space from 3.5MB/image to 2MB/image,
# but requires recalculating every kp and descriptor at the end of the offline fase (+45% runtime)
SAVE_DISK_SPACE_DURING_RUNNING = False


# How much RAM/memory is available for the program.
# In pycharm you can increase this under Help>Change memory settings.
# In a window machine, you can increase the cache size by following the steps on:
# https://www.windowscentral.com/how-change-virtual-memory-size-windows-10
#
# if possible provide (at least) '2 x size of all des' as available memory.
# if not enough ram
MAX_RAM_SIZE_GB = 60

# The amount of images returned in online.initial_scoring()
NB_OF_IMAGES_CONSIDERED = 5

# MORE ADVANCED PARAMETERS

# if RECALC_DES = False, the pre-calculated descriptors won't be calculated again if the file already exists.
RECALC_DES = False

# ONLY SET FALSE IF IMAGES ARE ALREADY RESIZED AND GRAYSCALED
RESIZE_IMAGES = False

# This permanently stores the keypoints and descriptors in the folder calc/
# This requires a lot of disk space, roughly x10 the image size, but drastically reduces the online lookup time.
PRE_CALC_DES = False

# delete data/ folder :(
# names.p.p should be defined
# make sure you have a backup of data/!
SAVE_EVEN_MORE_SPACE = True

# Criteria used in the k-means algorithm. TODO Defines a large portion of how long building tree will take.
# Good for data set of 100_000 image. For smaller data sets definitely increase the TERM_CRITERIA_MAX_ITER and ATTEMPTS.
# docs: https://docs.opencv.org/4.5.2/d1/d5c/tutorial_py_kmeans_opencv.html
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1, 0.5)
ATTEMPTS = 1

# precision used in the tree. (These parameters should be fine)
# PRECISION_FLOATS is the precision used for the scores of every item. (f.e. np.float32 or np.float64)
# PRECISION_IDS should be able to represent the largest ID in your database. (f.e. uint16 or uint32 (no - ids, only +))
# PRECISION_COUNTER represents the amount of descriptors of a certain image id end up in a single leaf node.
# The more PRECISION, the more accurate the scores, but the larger the tree and thus also slower.
# TODO accuracy gain vs size/time
PRECISION_FLOATS = np.float32
PRECISION_LEAFS = "float32, object, object"  # change float32 according to the PRECISION_FLOATS variable (leave out np.)
PRECISION_IDS = np.uint32
PRECISION_IMAGES = "uint32, object"
PRECISION_COUNTER = np.uint32
