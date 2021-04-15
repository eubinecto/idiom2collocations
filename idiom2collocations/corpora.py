"""
Corpora to be used for training.
"""
import json
from typing import Generator, List


class Corpus:
    def __init__(self, train_ndjson_path: str):
        self.train_ndjson_path = train_ndjson_path

    def __iter__(self):
        raise NotImplementedError


class Coca(Corpus):
    def __iter__(self) -> Generator[List[str], None, None]:
        with open(self.train_ndjson_path, 'r') as fh:
            for line in fh:
                sents = json.loads(line)
                # this will be a list of lists.
                for sent in sents:
                    # at least two
                    if len(sent) > 1:
                        yield sent


class Opensub(Corpus):
    def __iter__(self) -> Generator[List[str], None, None]:
        with open(self.train_ndjson_path, 'r') as fh:
            for line in fh:
                sent = json.loads(line)
                yield sent
