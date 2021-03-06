from multiprocessing import Pool
from functools import partial
import resize
import db_image
import numpy as np

import utils


# helpers to use multiprocessing for certain functions
def parallelize_resize(paths):
    with Pool() as p:
        p.map(resize.resize_gray_store, paths)


def process_calc_img(id, weight=1):
    try:
        kp, des = db_image.DbImage(id).get_kp_des()
        if weight < 1:
            des = des[np.random.sample(len(des)) < weight]
        return np.array(id, dtype=np.uint32), utils.convert_kp_to_kpl(kp), des

    except Exception as e:
        print("Error at", id, str(e))
        return np.array(id, dtype=np.uint32), None, np.array([], dtype=np.float32)


def parallelize_calc(ids, weight=1):
    with Pool() as p:
        data = p.map(partial(process_calc_img, weight=weight), ids)
    return data
