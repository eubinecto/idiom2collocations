
from os import path
from pathlib import Path

# directories
HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
CORPORA_DIR = path.join(HOME_DIR, "corpora")  # this is always at home.
COCA_SPOK_DIR = path.join(CORPORA_DIR, "coca_spok")
COCA_MAG_DIR = path.join(CORPORA_DIR, "coca_mag")
COCA_FICT_DIR = path.join(CORPORA_DIR, "coca_fict")
PROJECT_DATA_DIR = path.join(DATA_DIR, "data_idiom2collocations")
PROJECT_LIB_DIR = str(Path(__file__).resolve().parent)
RESRCS_DIR = path.join(PROJECT_LIB_DIR, "resources")

# files - data
IDIOM2CONTEXT_DICT_BIN = path.join(PROJECT_DATA_DIR, "idiom2context_dict.bin")  # the dictionary to build..
IDIOM2CONTEXT_TSV = path.join(PROJECT_DATA_DIR, "idiom2context.tsv")  # the main one to build.
IDIOM2COLLS_TF_TSV = path.join(PROJECT_DATA_DIR, "idiom2colls_tf.tsv")  # with simple tf
IDIOM2COLLS_TFIDF_TSV = path.join(PROJECT_DATA_DIR, "idiom2colls_tfidf.tsv")  # with tfidf
IDIOM2COLLS_PMI_TSV = path.join(PROJECT_DATA_DIR, "idiom2colls_pmi.tsv")  # with point-wise mutual inclusive


# files - coca_spok
COCA_SPOK_TRAIN_NDJSON = path.join(COCA_SPOK_DIR, 'train.ndjson')

# files - coca_mag
COCA_MAG_TRAIN_NDJSON = path.join(COCA_MAG_DIR, 'train.ndjson')

# files - coca_fict
COCA_FICT_TRAIN_NDJSON = path.join(COCA_FICT_DIR, 'train.ndjson')

