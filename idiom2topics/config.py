
from pathlib import Path
from os import path
from elasticsearch_dsl import connections
from datetime import datetime


def now() -> str:
    """
    just for naming files.
    :return:
    """
    now_obj = datetime.now()
    return now_obj.strftime("%d_%m_%Y__%H_%M_%S")

now_str = now()

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
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR = path.join(RESULTS_DIR, 'idiom2topics_tfidf_sample')  # with tfidf
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_DIR = path.join(RESULTS_DIR, 'idiom2topics_lda_sample')  # with lda
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_DIR = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_DIR,
                                                         'attempt_{}'.format(now_str))
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TRAINED_DIR = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR,
                                                                  'trained')
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TRAINED_DIR = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_DIR,
                                                                'trained')


# paths to files
CORPUS_ORIGIN_MERGED_TXT = path.join(CORPUS_ORIGIN_MERGED_DIR, "OpenSubtitles.en-pt_br_en.txt")
CORPUS_ORIGIN_MERGED_REFINED_NDJSON = path.join(CORPUS_ORIGIN_MERGED_DIR, "OpenSubtitles.en-pt_br_en_refined.ndjson")
CORPUS_ORIGIN_SPLITS_FS_MANIFEST_CSV = path.join(CORPUS_ORIGIN_SPLITS_DIR, 'fs_manifest.csv')
CORPUS_SAMPLE_MERGED_TXT = path.join(CORPUS_SAMPLE_MERGED_DIR, 'OpenSubtitles.en-pt_br_en_sample.txt')
CORPUS_SAMPLE_MERGED_REFINED_NDJSON = path.join(CORPUS_SAMPLE_MERGED_DIR,
                                                "OpenSubtitles.en-pt_br_en_refined_sample.ndjson")
CORPUS_SAMPLE_SPLITS_FS_MANIFEST_CSV = path.join(CORPUS_SAMPLE_SPLITS_DIR, 'fs_manifest.csv')
CORPUS_SAMPLE_SPLITS_REFINED_FS_MANIFEST_CSV = path.join(CORPUS_SAMPLE_SPLITS_REFINED_DIR, 'fs_manifest.csv')

# results
RESULTS_SAMPLE_IDIOM2CONTEXT_TSV = path.join(RESULTS_DIR, 'idiom2context_sample.tsv')
RESULTS_SAMPLE_IDIOM2TOPICS_TF_TSV = path.join(RESULTS_DIR,
                                               'idiom2topics_tf_sample.tsv')  # with tf
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR,
                                                  'idiom2topics_tfidf_sample.tsv')  # with tfidf
RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_MODEL = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR,
                                                    'model')  # tfidf, with the model.
# for LDA.
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TSV = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_DIR,
                                                        'idiom2topics_lda_sample.tsv')  # with lda
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_MODEL = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TRAINED_DIR,
                                                          'model')  # with lda
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TOPICS_TSV = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_DIR,
                                                               'topics.tsv')  # with lda
RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_META_JSON = path.join(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_DIR,
                                                              'meta.json')
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
