
from pathlib import Path
from os import path

# directories
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
LIB_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(PROJECT_ROOT_DIR, "data")
MOVIES_CORPUS_DIR = path.join(DATA_DIR, "movies_corpus")
OPENSUB_CORPUS_DIR = path.join(DATA_DIR, "opensub_corpus")

# path to files
OPENSUB_CORPUS_TXT_PATH = path.join(OPENSUB_CORPUS_DIR, "OpenSubtitles.en-pt_br_en.txt")
OPENSUB_CORPUS_WORDCNT_PATH = path.join(OPENSUB_CORPUS_DIR, "word_count")
MOVIES_CORPUS_TXT_PATH = path.join(MOVIES_CORPUS_DIR, "movies_text.txt")


# elasticsearch configs






