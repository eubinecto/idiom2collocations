import csv
import json
from typing import Generator, List, Tuple
csv.field_size_limit(100000000)


class Idiom2ContextLoader:
    def __init__(self, idiom2contexts_tsv_path: str):
        self.idiom2contexts_tsv_path = idiom2contexts_tsv_path

    def load(self) -> Generator[Tuple[str, List[str]], None, None]:
        with open(self.idiom2contexts_tsv_path, 'r') as fh:
            tsv_reader = csv.reader(fh, delimiter="\t")
            for row in tsv_reader:
                yield row[0], json.loads(row[1])
