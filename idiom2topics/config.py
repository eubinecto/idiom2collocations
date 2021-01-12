
from pathlib import Path
from os import path, environ

# directories
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
LIB_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(PROJECT_ROOT_DIR, "data")
MOVIES_CORPUS_DIR = path.join(DATA_DIR, "movies_corpus")
OPENSUB_CORPUS_DIR = path.join(DATA_DIR, "opensub_corpus")
# to be used by pyspark.
OPENSUB_CORPUS_WORDCNT_DIR = path.join(OPENSUB_CORPUS_DIR, "word_count")
OPENSUB_CORPUS_REFINED_DIR = path.join(OPENSUB_CORPUS_DIR, "refined")

# path to files
OPENSUB_CORPUS_TXT_PATH = path.join(OPENSUB_CORPUS_DIR, "OpenSubtitles.en-pt_br_en.txt")
OPENSUB_CORPUS_REFINED_TXT_PATH = path.join(OPENSUB_CORPUS_REFINED_DIR, "refined.txt")
MOVIES_CORPUS_TXT_PATH = path.join(MOVIES_CORPUS_DIR, "movies_text.txt")


# elasticsearch configs


# so that I can access this in a shell script
environ['OPENSUB_CORPUS_REFINED_DIR'] = OPENSUB_CORPUS_REFINED_DIR
environ['OPENSUB_CORPUS_REFINED_TXT_PATH'] = OPENSUB_CORPUS_REFINED_TXT_PATH



