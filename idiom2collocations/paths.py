
from os import path
from pathlib import Path
import sys
import csv
# very huge csv files.
csv.field_size_limit(sys.maxsize)


# directories
HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
CORPORA_DIR = path.join(HOME_DIR, "corpora")  # this is always at home.
COCA_SPOK_DIR = path.join(CORPORA_DIR, "coca_spok")
COCA_MAG_DIR = path.join(CORPORA_DIR, "coca_mag")
COCA_FICT_DIR = path.join(CORPORA_DIR, "coca_fict")
OPENSUB_DIR = path.join(CORPORA_DIR, "opensub")
PROJECT_DATA_DIR = path.join(DATA_DIR, "data_idiom2collocations")
PROJECT_LIB_DIR = str(Path(__file__).resolve().parent)
RESRCS_DIR = path.join(PROJECT_LIB_DIR, "resources")
IDIOMS_DIR = path.join(CORPORA_DIR, "idioms")


# files - idiom_keys
IDIOM2SENT_TSV = path.join(IDIOMS_DIR, "idiom2sent.tsv")  # just raw sentences, wherein the idiom appears in.
IDIOM2BOWS_TSV = path.join(IDIOMS_DIR, "idiom2bows.tsv")  # lemmatised. cleaned. stopwords may have been removed. depends.
IDIOM2LEMMA2POS_TSV = path.join(IDIOMS_DIR, "idiom2lemma2pos.tsv")  # lemmatised, cleaned. stopwords are removed.
IDIOM2LEMMA2POS_REMOVED_TSV = path.join(IDIOMS_DIR, "idiom2lemma2pos_removed.tsv")  # lemmatised, cleaned. stopwords are removed.
IDIOM2FREQ_TSV = path.join(IDIOMS_DIR, "idiom2freq.tsv")  # idiom, occurrences in the corpus.

# files - data. The extracted collocations will be stored here.
IDIOM2COLLS_TF_TSV = path.join(PROJECT_DATA_DIR, "idiom2colls_tf.tsv")  # with point-wise mutual inclusive.
IDIOM2COLLS_TFIDF_TSV = path.join(PROJECT_DATA_DIR, "idiom2colls_tfidf.tsv")  # with point-wise mutual inclusive.
IDIOM2COLLS_PMI_TSV = path.join(PROJECT_DATA_DIR, "idiom2colls_pmi.tsv")  # with point-wise mutual inclusive.
LEMMA2IDFS_TSV = path.join(PROJECT_DATA_DIR, "lemma2idfs.tsv")

# files - coca_spok
COCA_SPOK_TRAIN_NDJSON = path.join(COCA_SPOK_DIR, 'train.ndjson')

# files - coca_mag - may not be able to use them.
COCA_MAG_TRAIN_NDJSON = path.join(COCA_MAG_DIR, 'train.ndjson')

# files - opensub
OPENSUB_TRAIN_NDJSON = path.join(OPENSUB_DIR, 'train.ndjson')

# spacy
NLP_MODEL = "en_core_web_sm"
