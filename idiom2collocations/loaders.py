import json
from functional import seq
from functional.pipeline import Sequence
from idiom2collocations.paths import IDIOM2SENT_TSV, IDIOM2CLUSTERS_TSV, IDIOM2LEMMA2POS_TSV


def load_idiom2sent() -> Sequence:
    """
    idiom, sent (list of tokens)
    """
    return seq.csv(IDIOM2SENT_TSV, delimiter="\t") \
              .map(lambda row: (row[0], json.loads(row[1])))


def load_idiom2lemma2pos() -> Sequence:
    return seq.csv(IDIOM2LEMMA2POS_TSV, delimiter="\t") \
        .map(lambda row: (row[0], json.loads(row[1])))


def load_idiom2clusters() -> Sequence:
    return seq.csv(IDIOM2CLUSTERS_TSV, delimiter="\t") \
        .map(lambda row: (row[0],
                          json.loads(row[1]),
                          json.loads(row[2]),
                          json.loads(row[3]),
                          json.loads(row[4]),
                          json.loads(row[5]),
                          json.loads(row[6])))
