import json
from functional import seq
from functional.pipeline import Sequence
from idiom2collocations.paths import IDIOM2SENT_TSV, IDIOM2LEMMA2POS_TSV, IDIOM2BOWS_TSV,\
    IDIOM2COLLS_TF_TSV, IDIOM2COLLS_TFIDF_TSV, IDIOM2COLLS_PMI_TSV


def load_idiom2sent() -> Sequence:
    """
    idiom, sent (list of tokens)
    """
    return seq.csv(IDIOM2SENT_TSV, delimiter="\t") \
              .map(lambda row: (row[0], json.loads(row[1])))


def load_idiom2lemma2pos() -> Sequence:
    return seq.csv(IDIOM2LEMMA2POS_TSV, delimiter="\t") \
              .map(lambda row: (row[0], json.loads(row[1])))


def load_idiom2bows() -> Sequence:
    return seq.csv(IDIOM2BOWS_TSV, delimiter="\t") \
              .map(lambda row: (row[0],
                   json.loads(row[1]),
                   json.loads(row[2]),
                   json.loads(row[3]),
                   json.loads(row[4])))


def load_idiom2colls(mode: str) -> Sequence:
    if mode == "tf":
        colls_tsv_path = IDIOM2COLLS_TF_TSV
    elif mode == "tfidf":
        colls_tsv_path = IDIOM2COLLS_TFIDF_TSV
    elif mode == "pmi":
        colls_tsv_path = IDIOM2COLLS_PMI_TSV
    else:
        raise ValueError
    return seq.csv(colls_tsv_path, delimiter="\t") \
        .map(lambda row: (row[0],
                          json.loads(row[1]),  # verb colls
                          json.loads(row[2]),  # noun colls
                          json.loads(row[3]),  # adj colls
                          json.loads(row[4])))  # adv colls
