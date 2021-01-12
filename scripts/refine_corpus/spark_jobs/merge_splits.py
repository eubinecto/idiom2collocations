import re
from fsplit.filesplit import Filesplit
from idiom2topics.config import\
    OPENSUB_CORPUS_SPLITS_REFINED_MANIFEST_CSV_PATH,\
    OPENSUB_CORPUS_SPLITS_REFINED_DIR,\
    SPLIT_SIZE,\
    OPENSUB_CORPUS_REFINED_NDJSON_PATH
import csv
import os


def generate_manifest():
    idx_re = re.compile(r"OpenSubtitles.en-pt_br_en_(.*)_refined.ndjson")
    # get all the split files and sort them
    split_names = sorted([
        file_name
        for file_name in os.listdir(OPENSUB_CORPUS_SPLITS_REFINED_DIR)
        if idx_re.match(file_name)
    ], key=lambda x: int(idx_re.findall(x)[0]))

    schema = "filename,filesize,encoding,header".split(",")
    with open(OPENSUB_CORPUS_SPLITS_REFINED_MANIFEST_CSV_PATH, 'w') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(schema)
        for split_name in split_names:
            to_write = [split_name, str(SPLIT_SIZE), "", ""]
            csv_writer.writerow(to_write)


def main():
    # sort the files, and merge them into one.
    generate_manifest()
    fs = Filesplit()
    fs.merge(input_dir=OPENSUB_CORPUS_SPLITS_REFINED_DIR,
             output_file=OPENSUB_CORPUS_REFINED_NDJSON_PATH)


if __name__ == '__main__':
    main()
