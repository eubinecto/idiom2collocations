
from pathlib import Path
from os import path

# directories
from merge_idioms.builders import MIPBuilder
from spacy import Language

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
LIB_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(PROJECT_ROOT_DIR, "data")
OPENSUB_CORPUS_DIR = path.join(DATA_DIR, "opensub_corpus")

# to be used by pyspark.
OPENSUB_CORPUS_SPLITS_DIR = path.join(OPENSUB_CORPUS_DIR, "splits")
OPENSUB_CORPUS_SPLITS_REFINED_DIR = path.join(OPENSUB_CORPUS_DIR, "splits_refined")
OPENSUB_CORPUS_SPLITS_EXAMPLES_DIR = path.join(OPENSUB_CORPUS_DIR, "splits_examples")

# path to files under OPENSUB_DIR
OPENSUB_CORPUS_TXT_PATH = path.join(OPENSUB_CORPUS_DIR, "OpenSubtitles.en-pt_br_en.txt")
OPENSUB_CORPUS_REFINED_NDJSON_PATH = path.join(OPENSUB_CORPUS_DIR, "OpenSubtitles.en-pt_br_en_refined.ndjson")
OPENSUB_CORPUS_EXAMPLES_NDJSON_PATH = path.join(OPENSUB_CORPUS_DIR, "OpenSubtitles.en-pt_br_en_examples.ndjson")

# path to files under REFINED_DIR
OPENSUB_CORPUS_SPLITS_REFINED_MANIFEST_CSV_PATH = path.join(OPENSUB_CORPUS_SPLITS_REFINED_DIR, 'fs_manifest.csv')

# split size
SPLIT_SIZE = 900000


# configs for refining corpus & building examples
NUM_PROCS = 4
MIN_LENGTH = 2
NUM_CONTEXTS = 10