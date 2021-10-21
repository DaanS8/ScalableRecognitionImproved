"""
Change the parameters of the program here.
"""
# define max image size, if images are bigger, image is rescaled so that the width and height are <= MAX_IMAGE_SIZE.
# A larger MAX_IMAGE_SIZE is more accurate but slower and requires more storage.
MAX_IMAGE_SIZE = 1080

# On https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors benchmarks of the faiss library of different
# database structures using different vectors are given, under the menu item "Typical use cases and benchmarks".
FACTORY_INDEX = "OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr"


# The number _,IVFxxxxxx_,_ defines how large your training set needs to be.
# The number of descriptors used in the training set lie between 30*xxxxxx` and `256*xxxxxx.
# Set this paramter equal to #training_des/#total_des.
# The more #training_des you use, the accurate the index, but takes longer to train.
FRACTION_DES_USED_FOR_TRAINING = 0.05

IndexFileName = "index_added_final.f"
KpListName = "kp_lookup_new.f"


# The amount of images returned in online.initial_scoring()
NB_OF_IMAGES_CONSIDERED = 10

# ONLY SET FALSE IF IMAGES ARE ALREADY RESIZED AND GRAYSCALED
RESIZE_IMAGES = False
