from idiom2topics.config import OPENSUB_CORPUS_TXT_PATH, OPENSUB_CORPUS_SPLITS_DIR, SPLIT_SIZE
from fsplit.filesplit import Filesplit


def split_cb(f, s):
    print("file: {0}, size: {1}".format(f, s))


def main():
    # first.. count the number of lines
    fs = Filesplit()
    fs.split(file=OPENSUB_CORPUS_TXT_PATH,
             split_size=SPLIT_SIZE, output_dir=OPENSUB_CORPUS_SPLITS_DIR, callback=split_cb)


if __name__ == '__main__':
    main()
