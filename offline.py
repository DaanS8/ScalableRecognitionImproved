import os

import faiss
import parallelize
import cv2 as cv
import numpy as np
from multiprocessing import Pool
from functools import partial
import db_image
import utils
from parameters import *


def train_index(index):
    print("Getting descriptors for training index.")
    data = parallelize.parallelize_calc(db_image.get_ids(), weight=FRACTION_DES_USED_FOR_TRAINING)
    if data is not None:
        print("Post-processing descriptors.")
        des = list()
        for _, _, d in data:
            if d is not None and 0 != np.size(d, axis=0):
                des.append(d)

        print("Concatenating descriptors.")
        des = np.concatenate(des, axis=0, dtype=np.float32)

        print("Start training index.")
        index.train(des)

        print("Training finished! Storing index.")
        faiss.write_index(index, IndexFileName)


def adding_to_index():
    print("Load index.")
    index = faiss.read_index(IndexFileName)  # GPU index_ivf might screw adding to index up? Load from file to be sure.

    print("Start adding descriptors to index.")
    kp_list, counter = [], len(db_image.get_ids())
    for image_id in db_image.get_ids():
        image_id = int(image_id)
        kp_image, des_image = db_image.DbImage(image_id).get_kp_des()
        for kp in kp_image:
            kp_list.append((kp.pt, image_id))
        index.add(des_image)
        counter -= 1
        if counter % 1000 == 0:
            print("{:_} images left to add.".format(counter))

    print("All des added to index! Post-processing kp_list.")
    kp_list = np.array(kp_list, dtype=object)

    print("Storing keypoint list.")
    utils.pickle_data(kp_list, KpListName)

    print("Storing final index.")
    faiss.write_index(index, IndexFileName)


def main():
    if RESIZE_IMAGES:
        print("Resizing and grayscaling database.")
        paths = ["data/" + path for path in os.listdir("data/")]
        parallelize.parallelize_resize(paths)

    print("Setting up index.")
    index = faiss.index_factory(128, FACTORY_INDEX)
    res = faiss.StandardGpuResources()  # use a single GPU

    # Use GPU for training
    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(64))
    index_ivf.clustering_index = clustering_index

    train_index(index)
    adding_to_index()


if __name__ == "__main__":
    main()
