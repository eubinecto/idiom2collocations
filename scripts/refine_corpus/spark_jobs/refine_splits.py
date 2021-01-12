# TODO: in merge_idioms, having load_mip() would be nice
from datetime import datetime
from typing import Tuple, List
import os
import multiprocessing as mp
from idiom2topics.config import\
    OPENSUB_CORPUS_SPLITS_DIR,\
    OPENSUB_CORPUS_SPLITS_REFINED_DIR,\
    NUM_PROCS,\
    MIP
import json





# to log everything. I'll put date and time to everything.
def now() -> str:
    now_obj = datetime.now()
    return now_obj.strftime("%d_%m_%Y__%H_%M_%S")


def to_refined(zipped_path: Tuple[str, str]):
    split_path, out_path = zipped_path
    with open(split_path, 'r') as s_fh, open(out_path, 'w') as o_fh:
        for line in s_fh:
            refined_tokens = refine(line.strip())
            o_fh.write(json.dumps(refined_tokens) + "\n")
    print("{}:DONE:split_path={}, out_path={}"
          .format(now(),split_path, out_path))


def main():
    # get the paths to all splits
    split_names = [
        split_name
        for split_name in os.listdir(OPENSUB_CORPUS_SPLITS_DIR)
        if split_name.endswith(".txt")
    ]

    split_paths = [
        os.path.join(OPENSUB_CORPUS_SPLITS_DIR, split_name)
        for split_name in split_names
    ]

    # create output paths for each split
    out_paths = [
        os.path.join(OPENSUB_CORPUS_SPLITS_REFINED_DIR, split_name.replace(".txt", '_refined.ndjson'))
        for split_name in split_names
    ]

    zipped_paths = zip(split_paths, out_paths)

    with mp.Pool(processes=NUM_PROCS) as p:
        # async map, map
        # https://m.blog.naver.com/PostView.nhn?blogId=parkjy76&logNo=221089918474&proxyReferer=https:%2F%2Fwww.google.com%2F
        r = p.map_async(to_refined, zipped_paths)
        r.wait()


if __name__ == '__main__':
    main()
