from importlib import reload
import utils
import time
import faiss
import online_debug

fileNameIndex = "index_added_final.f"
kpDbName = "kp_lookup_new.f"

def container():
    start = time.time()
    kp_db = utils.get_pickled(kpDbName)
    index = faiss.read_index(fileNameIndex)
    print("loading the database took {}s.".format(int(time.time() - start)))
    while True:
        if input("press enter to restart") == "":
            try:
                reload(online_debug)
                from online_debug import main
                main(kp_db, index)
            except Exception as e:
                print("Error", e)


if __name__ == "__main__":
    container()