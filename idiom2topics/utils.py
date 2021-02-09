import json
from typing import List
from functional import seq
from functional.pipeline import Sequence

from config import RESULTS_SAMPLE_IDIOM2CONTEXT_TSV


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
