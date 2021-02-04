
from pathlib import Path
from os import path
from elasticsearch_dsl import connections
# directories
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
LIB_ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(PROJECT_ROOT_DIR, "data")
CORPUS_DIR = path.join(DATA_DIR, "opensub_corpus")
CORPUS_ORIGIN_DIR = path.join(CORPUS_DIR, "origin")
CORPUS_ORIGIN_MERGED_DIR = path.join(CORPUS_ORIGIN_DIR, 'merged')
CORPUS_ORIGIN_SPLITS_DIR = path.join(CORPUS_ORIGIN_DIR, 'splits')
CORPUS_ORIGIN_SPLITS_REFINED_DIR = path.join(CORPUS_ORIGIN_DIR, 'splits_refined')
CORPUS_SAMPLE_DIR = path.join(CORPUS_DIR, 'sample')
CORPUS_SAMPLE_MERGED_DIR = path.join(CORPUS_SAMPLE_DIR, 'merged')
CORPUS_SAMPLE_SPLITS_DIR = path.join(CORPUS_SAMPLE_DIR, 'splits')
CORPUS_SAMPLE_SPLITS_REFINED_DIR = path.join(CORPUS_SAMPLE_DIR, 'splits_refined')
RESULTS_DIR = path.join(DATA_DIR, 'results')
RESULTS_SAMPLE_IDIOM2TOPICS_TF_DIR = path.join(RESULTS_DIR, 'idiom2topics_tf_sample')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR = path.join(RESULTS_DIR, 'idiom2topics_tfidf_sample')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_DIR = path.join(RESULTS_DIR, 'idiom2topics_lda_sample')  #
RESULTS_SAMPLE_IDIOM2TOPICS_TF_SPLITS_DIR = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_TF_DIR, 'splits')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_SPLITS_DIR = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR, 'splits')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_SPLITS_DIR = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_DIR, 'splits')  #
# paths to files
CORPUS_ORIGIN_MERGED_TXT = path.join(CORPUS_ORIGIN_MERGED_DIR, "OpenSubtitles.en-pt_br_en.txt")
CORPUS_ORIGIN_MERGED_REFINED_NDJSON = path.join(CORPUS_ORIGIN_MERGED_DIR, "OpenSubtitles.en-pt_br_en_refined.ndjson")
CORPUS_ORIGIN_SPLITS_FS_MANIFEST_CSV = path.join(CORPUS_ORIGIN_SPLITS_DIR, 'fs_manifest.csv')
CORPUS_SAMPLE_MERGED_TXT = path.join(CORPUS_SAMPLE_MERGED_DIR, 'OpenSubtitles.en-pt_br_en_sample.txt')
CORPUS_SAMPLE_MERGED_REFINED_NDJSON = path.join(CORPUS_SAMPLE_MERGED_DIR,
                                                "OpenSubtitles.en-pt_br_en_refined_sample.ndjson")
CORPUS_SAMPLE_SPLITS_FS_MANIFEST_CSV = path.join(CORPUS_SAMPLE_SPLITS_DIR, 'fs_manifest.csv')
CORPUS_SAMPLE_SPLITS_REFINED_FS_MANIFEST_CSV = path.join(CORPUS_SAMPLE_SPLITS_REFINED_DIR, 'fs_manifest.csv')
RESULTS_SAMPLE_IDIOM2CONTEXT_TSV = path.join(RESULTS_DIR, 'idiom2context_sample.tsv')
RESULTS_SAMPLE_IDIOM2TOPICS_TF_TSV = path.join(RESULTS_DIR, 'idiom2topics_tf_sample.tsv')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV = path.join(RESULTS_DIR, 'idiom2topics_tfidf_sample.tsv')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_TSV = path.join(RESULTS_DIR, 'idiom2topics_lda_sample.tsv')  # with lda technique

# file split
SPLIT_SIZE = 900000

# schemas
FS_MANIFEST_CSV_SCHEMA = "filename,filesize,encoding,header".split(",")

# configs for mp
NUM_PROCS = 4

# ES configs
HOST = 'localhost'
ES_HOST = '{}:9200'.format(HOST)
ES_CLIENT = connections.create_connection(hosts=ES_HOST)
BATCH_SIZE = 10000
