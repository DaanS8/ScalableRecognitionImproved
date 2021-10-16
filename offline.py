import faiss
import parallelize
import cv2 as cv
import numpy as np
from multiprocessing import Pool
from functools import partial
import db_image
import utils


def concat(arr):
    return np.concatenate(arr, axis=0, dtype=np.float32)


def main():
    index = faiss.index_factory(128, "OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr")
    res = faiss.StandardGpuResources()  # use a single GPU

    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(64))
    index_ivf.clustering_index = clustering_index

    data = parallelize.parallelize_calc(db_image.get_ids(), weight=0.05)
    if data is not None:
        ids, des = list(), list()
        for i, _, d in data:
            if d is not None and 0 != np.size(d, axis=0):
                ids.extend([i] * np.size(d, axis=0))
                des.append(d)
        utils.pickle_data(ids, "counter.f")
        des = concat(des)

        index.train(des)  # failes here
        faiss.write_index(index, "index2.f")


if __name__ == "__main__":
    main()
