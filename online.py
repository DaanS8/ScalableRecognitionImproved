import db_image
import resize
import os
import utils
import numpy as np
import time
import cv2 as cv
import faiss
from parameters import *


def final_scoring(results):
    good = dict()

    # Decrease the amount of matches needed until at least 5 image candidates are found
    minimum_matches = NB_OF_MATCHES_CONSIDERED
    counter = sum([1 if len(tmp) > minimum_matches else 0 for tmp, _ in results.values()])
    while minimum_matches > 5 > counter:
        minimum_matches -= 1
        counter = sum([1 if len(tmp) > minimum_matches else 0 for tmp, _ in results.values()])

    # For the (at least) 5 best images, try geometric verification
    for k, v in results.items():
        db, matches = v
        if len(db) > minimum_matches:
            # Get x, y values
            src_pts = np.float32(db).reshape(-1, 1, 2)
            dst_pts = np.float32([m.pt for m in matches]).reshape(-1, 1, 2)

            # Calculate Homography
            _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            # Count nb of inliers
            final = np.count_nonzero(mask)

            # If at least 6 inliers, good result!
            if final > 5:
                good[k] = final
    return good


def main():
    print("Loading index and kp list")
    start = time.time()
    kp_db = utils.get_pickled(KpListName)
    index = faiss.read_index(IndexFileName)
    print("loading the database took {}s.".format(int(time.time() - start)))

    while True:
        # Get an image path to process
        image_path = ""
        while (not os.path.isfile(image_path)) and image_path != "q":
            image_path = input("Give path to input image (jpg), enter q to quit:")
            if image_path[-4:] != ".jpg" and image_path != "q":
                image_path += ".jpg"

        # Check if exit is needed
        print("")
        if image_path == "q":
            print("Exiting.")
            break

        # Start timer
        start = time.time()

        # Get grayscaled image
        img = resize.get_resize_gray(image_path)

        # Calculate kp and des
        kp, des = utils.sift.detectAndCompute(img, None)

        # Search index
        _, I = index.search(des, K)

        # Process results
        results = dict()
        for kp_i, indices in zip(kp, I):
            kps_db = kp_db[indices]
            for i, k in zip(indices, kps_db):
                id = int(k[1])
                db, match = results.get(id, [[], []])
                db.append(k[0])
                match.append(kp_i)
                results[id] = db, match

        # Final scoring
        good = final_scoring(results)

        if len(good) > 0:  # any results found?
            # Sorting result
            final_result = {k: v for k, v in sorted(good.items(), key=lambda item: item[1], reverse=True)}

            # Start calculating certainty scores
            minimal_value = max(final_result.values()) * 0.2
            good = {k: v for k, v in final_result.items() if v >= minimal_value}
            sum_values = sum(good.values())

            first = True
            for key, value in sorted(good.items(), key=lambda item: item[1], reverse=True):
                certainty_percentage = min(100, value - 5) * value / sum_values

                # Check if the best result is certain/uncertain
                if first:
                    first = False
                    if sum_values < 105 or certainty_percentage < 50:
                        print("Uncertain results, please consider taking a new picture.")

                # lookup name of the result
                name = db_image.DbImage(key).get_name()
                if name == "":
                    name = str(key)

                # result
                print("{}: {:.2f}%".format(name, certainty_percentage))
        else:
            print("No result found, please take a new picture.")

        print("Processing this result took {:.2f}s.".format(time.time() - start))
        print("")


if __name__ == "__main__":
    main()

