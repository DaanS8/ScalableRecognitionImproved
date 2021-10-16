import time

import faiss
import db_image
import resize
import utils
import numpy as np
import cv2 as cv
import os
from collections import OrderedDict
import parallelize
import heapq

fileNameIndex = "index_added_final.f"
kpDbName = "kp_lookup_new.f"
inputImage = "200890.jpg"
test_folder = "testset/"
K = 10
NB_ITEMS_CONSIDERED = 5


def _look_for_sub_test_folders(test_folder):
    # Check if there are sub test folders in the given test_folder
    # If there are sub folders, use these as test_folders
    # If there are no sub folders, use the given test_folder in test_folders
    try:
        test_folders = [test_folder + name_folder + "/" for name_folder in os.listdir(test_folder)]
        if len(test_folders) == 0:
            test_folders = [test_folder]
    except Exception:  # only a main folder
        test_folders = [test_folder]
    return test_folders


def load_index_db():
    kp_db = utils.get_pickled(kpDbName)
    index = faiss.read_index(fileNameIndex)
    return kp_db, index


def main(kp_db, index):
    print("Length of kp_db {:_}. Length of index {:_}. Equal? {}."
          .format(len(kp_db), index.ntotal, str(len(kp_db) == index.ntotal)))
    folders = _look_for_sub_test_folders(test_folder)

    start_total = time.time()
    t0, t1, t2, t3, t4, t5, t6 = 0, 0, 0, 0, 0, 0, 0
    processed_images = 0
    for folder in folders:
        result = dict()
        percentage_correct = dict()
        percentage_incorrect = dict()
        certain = [0, 0]
        uncertain = [0, 0]
        no_result = [0, 0]

        start = time.time()
        paths = [folder + file_name for file_name in os.listdir(folder)]
        parallelize.parallelize_resize(paths)
        t0 += time.time() - start

        processed_images += len(paths)
        print("Start processing folder '{}' with index '{}'.".format(folder, fileNameIndex))
        for path in paths:
            #try:
            start = time.time()
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            kp, des = utils.sift.detectAndCompute(img, None)

            t1 += time.time() - start

            start = time.time()
            D, I = index.search(des, K)
            t2 += time.time() - start

            start = time.time()
            results = dict()
            for kp_i, indices, distances in zip(kp, I, D):
                kps_db = kp_db[indices]
                for i, k, d in zip(indices, kps_db, distances):
                    id = int(k[1])
                    db, match = results.get(id, [[], []])
                    db.append(k[0])
                    match.append(kp_i)
                    results[id] = db, match
            t3 += time.time() - start

            start = time.time()
            if "Junk" not in path:
                correct_id = int(path[path.rfind("/") + 1:-4])
            else:
                correct_id = -1

            sort = {k: len(v[0]) for k, v in sorted(results.items(), key=lambda item: len(item[1][0]), reverse=True)}

            correct_index = -1
            if correct_id != -1:
                for i, key in enumerate(sort.keys()):
                    if key == correct_id:
                        correct_index = i
                        break
            result[correct_index] = result.get(correct_index, 0) + 1
            t4 += time.time() - start

            start = time.time()
            good = dict()

            minimum_items = 10
            counter = sum([1 if len(tmp) > minimum_items else 0 for tmp, _ in results.values()])
            while minimum_items > 5 > counter:
                minimum_items -= 1
                counter = sum([1 if len(tmp) > minimum_items else 0 for tmp, _ in results.values()])

            for k, v in results.items():
                db, matches = v
                if len(db) > minimum_items:
                    src_pts = np.float32(db).reshape(-1, 1, 2)
                    dst_pts = np.float32([m.pt for m in matches]).reshape(-1, 1, 2)
                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    final = np.count_nonzero(mask)
                    if final > 5:
                        good[k] = final
            t5 += time.time() - start

            start = time.time()
            if len(good) > 0:
                final_result = {k: v for k, v in sorted(good.items(), key=lambda item: item[1], reverse=True)}

                minimal_value = max(final_result.values()) * 0.2
                good = {k: v for k, v in final_result.items() if v >= minimal_value}
                sum_values = sum(good.values())
                first = True
                for key, value in sorted(good.items(), key=lambda item: item[1], reverse=True):
                    certainty_percentage = min(100, value - 5) * value / sum_values
                    if first:
                        first = False
                        if key == correct_id:
                            percentage_correct[certainty_percentage] = \
                                percentage_correct.get(certainty_percentage, 0) + 1
                        else:
                            percentage_incorrect[certainty_percentage] = percentage_incorrect.get(
                                certainty_percentage, 0) + 1

                        # Certain/Uncertain as boolean value
                        if sum_values < 105 or certainty_percentage < 50:
                            if key == correct_id:
                                uncertain[0] += 1
                            else:
                                uncertain[1] += 1
                        else:
                            if key == correct_id:
                                certain[0] += 1
                            else:
                                certain[1] += 1
            else:
                if correct_index == -1:
                    no_result[0] += 1
                else:
                    no_result[1] += 1
            t6 += time.time() - start
            #except Exception as e:
                #print("Error at {} with error: {}".format(path, e))
        print("position index", OrderedDict(sorted(result.items())))
        print("Certainty percentage correct", OrderedDict(sorted(percentage_correct.items())))
        print("Certainty percentage incorrect", OrderedDict(sorted(percentage_incorrect.items())))
        print("Certain", certain)
        print("Uncertain", uncertain)
        print("No Result", no_result)

    print("Average time per image: {:.2f}s.".format((time.time() - start_total)/processed_images))

    sum_times = (t0 + t1 + t2 + t3 + t4 + t5 + t6) / 100
    t0, t1, t2, t3, t4, t5, t6 = \
        t0 / sum_times, t1 / sum_times, t2 / sum_times, t3 / sum_times, t4 / sum_times, t5 / sum_times, t6 / sum_times
    print("Percentages: Preprocessing images {:.2f}, Calculating kp & des {:.2f}, Searching Index {:.2f}, "
          "Processing search {:.2f}, Accuracy Calculations {:.2f},Geometric Verification {:.2f}, Certainty Calculation {:.2f}."
          .format(t0, t1, t2, t3, t4, t5, t6))

if __name__ == "__main__":
    start = time.time()
    kp_db, index = load_index_db()
    print("loading the database took {}s.".format(int(time.time() - start)))
    main(kp_db, index)
