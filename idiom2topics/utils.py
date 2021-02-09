import json
from typing import List, Dict
from functional import seq
from functional.pipeline import Sequence

from config import RESULTS_SAMPLE_IDIOM2CONTEXT_TSV, \
                   RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_NDJSON
from data import Idiom2Tfidfs


def flatten_to_tokens(contexts: List[List[str]]) -> List[str]:
    return [
        token
        for context in contexts
        for token in context
    ]


def load_idiom2context() -> Sequence:
    return seq.csv(RESULTS_SAMPLE_IDIOM2CONTEXT_TSV, delimiter="\t") \
              .map(lambda row: (row[0], json.loads(row[1])))


def load_idiom2contexts() -> Sequence:
    return load_idiom2context()\
           .group_by_key()


def load_idiom2contexts_flattened() -> Sequence:
    return load_idiom2contexts() \
           .map(lambda row: (row[0], flatten_to_tokens(row[1])))


def load_idiom2tfidfs(top_n: int = 50) -> Idiom2Tfidfs:
    """
    you will have to search for them... so just do it.
    :param top_n: necessary to manage memory usage
    :return: (idiom -> (term -> tfidf))
    """
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_NDJSON, 'r') as fh:
        idiom2tfidfs = dict()
        for line in fh:
            doc: dict = json.loads(line)
            idiom: str = doc['idiom']
            tfidfs: List[dict] = doc['tfidfs']
            tfidfs_dict = dict()
            # only the top n
            for entry in tfidfs[:top_n]:
                tfidfs_dict[entry['term']] = entry['tfidf']
            idiom2tfidfs[idiom] = tfidfs_dict
    return Idiom2Tfidfs(idiom2tfidfs)
