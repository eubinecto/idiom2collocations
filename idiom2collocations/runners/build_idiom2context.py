"""
load from corpora, and then build...
"""
import argparse
import json

from idiom2collocations.corpus import Coca
from idiom2collocations.paths import (
    COCA_MAG_TRAIN_NDJSON,
    COCA_SPOK_TRAIN_NDJSON,
    IDIOM2CONTEXT_TSV
)
from identify_idioms.service import load_idioms
from itertools import chain
import time
import csv


def main():
    # --- idioms to search for --- #
    idioms = [
        idiom.replace(" ", "_")  # replace white space with an under bar
        for idiom in load_idioms()
    ]
    idioms = set(idioms)  # turn it into a set, because we are only interested in.. looking for them.

    # --- load the corpora --- #
    cock_spok = Coca(COCA_SPOK_TRAIN_NDJSON, doc_is_sent=True)
    coca_mag = Coca(COCA_MAG_TRAIN_NDJSON, doc_is_sent=True)
    # TODO: coca_fict.
    sentences = chain(cock_spok, coca_mag)  # chain the generators.

    # I want to check how long it would take just iterating over all of the sentences
    start = time.process_time()
    total = 0
    # --- start writing --- #
    with open(IDIOM2CONTEXT_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idx_sent, sent in enumerate(sentences):
            total += 1
            if len(sent) < 2:  # should be more than 2.
                continue
            for idx_token, token in enumerate(sent):
                if token in idioms:
                    idiom = token
                    context = sent[:idx_token] + sent[idx_token + 1:]  # just omit it.
                    to_write = [idiom, json.dumps(context)]
                    tsv_writer.writerow(to_write)
    end = time.process_time()
    print("It took {} minutes to iterate over {} sents".format(str((end - start) / 60), str(total)))


if __name__ == '__main__':
    main()
