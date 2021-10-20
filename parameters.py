"""
Change the parameters of the program here.
"""
# define max image size, if images are bigger, image is rescaled so that the width and height are <= MAX_IMAGE_SIZE.
# A larger MAX_IMAGE_SIZE is more accurate but slower and requires more storage.
MAX_IMAGE_SIZE = 1080

FRACTION_DES_USED_FOR_TRAINING = 0.05

IndexFileName = "index_added_final.f"
KpListName = "kp_lookup_new.f"

FACTORY_INDEX = "OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr"
# How much memory is available for the program.
# In a window machine, you can increase the memory size by following the steps on:
# https://www.windowscentral.com/how-change-virtual-memory-size-windows-10
#
# if possible provide TODO
MAX_MEMORY_SIZE_GB = 60

# The amount of images returned in online.initial_scoring()
NB_OF_IMAGES_CONSIDERED = 10

# ONLY SET FALSE IF IMAGES ARE ALREADY RESIZED AND GRAYSCALED
RESIZE_IMAGES = False
