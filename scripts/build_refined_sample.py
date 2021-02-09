import csv
import os
import multiprocessing as mp
import re
from typing import List
from fsplit.filesplit import Filesplit
from merge_idioms.service import build_mip
from idiom2topics.config import CORPUS_SAMPLE_SPLITS_DIR, CORPUS_SAMPLE_SPLITS_REFINED_DIR, NUM_PROCS, \
    CORPUS_SAMPLE_SPLITS_REFINED_FS_MANIFEST_CSV, FS_MANIFEST_CSV_SCHEMA, SPLIT_SIZE, \
    CORPUS_SAMPLE_MERGED_REFINED_NDJSON

# to be used globally
mip = build_mip()


def normalise_case(sent: str):
    """
    lower the sentence, but be mindful of "I".
    it should stay uppercase.
    :param sent:
    :return:
    """
    return sent.lower() \
               .replace("i ", "I ") \
               .replace(" i ", " I ") \
               .replace("i'm", "I'm") \
               .replace("i'll", "I'll") \
               .replace("i\'d", "I'd")


def write_refined(sent: str) -> List[str]:
    """
    1. tokenise
    2. cleanse
    3. lemmatize
    :param sent:
    :return:
    """
    global mip
    try:
        lemmas = [
            # lemmatize
            # escape hex character..
            # TODO: regexp-> special characters should all be escaped.
            token.lemma_.replace("%", "").replace("=", "")
            # tokenize
            for token in mip(normalise_case(sent))
            # cleanse
            # TODO: filter out '♪'. filter out special characters.
            if not token.is_punct
            if not token.like_num
            if not token.is_stop
        ]
    except ValueError as ve:
        return ["ERROR:sent={},err={}".format(sent, str(ve))]
    else:
        return lemmas


def build_splits_refined():
    split_paths = [
        os.path.join(CORPUS_SAMPLE_SPLITS_DIR, split_name)
        for split_name in os.listdir(CORPUS_SAMPLE_SPLITS_DIR)
        if split_name.endswith(".ndjson")
    ]
    # create output paths
    out_paths = [
        os.path.join(CORPUS_SAMPLE_SPLITS_REFINED_DIR, split_name.replace("refined", "pairs"))
        for split_name in os.listdir(CORPUS_SAMPLE_SPLITS_DIR)
        if split_name.endswith(".ndjson")
    ]
    zipped_paths = zip(split_paths, out_paths)

    # doing this with multiple processes.
    # if you ignore the previous context at all -> you can make this embarrassingly parallel.
    # markov assumption...
    with mp.Pool(processes=NUM_PROCS) as p:
        p.map(write_refined, zipped_paths)


def build_splits_refined_manifest():
    with open(CORPUS_SAMPLE_SPLITS_REFINED_FS_MANIFEST_CSV, 'w') as fh:
        csv_writer = csv.writer(fh)
        # write the schema
        csv_writer.writerow(FS_MANIFEST_CSV_SCHEMA)
        split_names = [
            split_name
            for split_name in os.listdir(CORPUS_SAMPLE_SPLITS_REFINED_DIR)
            if split_name.endswith("ndjson")
        ]
        split_names = sorted(split_names, key=lambda x: int(re.findall(r'([0-9]+)', x)[0]))
        for split_name in split_names:
            to_write = [split_name, SPLIT_SIZE, "", ""]
            csv_writer.writerow(to_write)


def merge_splits_refined():
    fs = Filesplit()
    fs.merge(input_dir=CORPUS_SAMPLE_SPLITS_REFINED_DIR,
             output_file=CORPUS_SAMPLE_MERGED_REFINED_NDJSON)


def main():
    build_splits_refined()
    build_splits_refined_manifest()
    build_splits_refined()


if __name__ == '__main__':
    main()
