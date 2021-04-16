"""
idiom2contexts -> raw, tokenized sents.
what if two idioms pos_co-occur? Didn't think of that, sadly.
first_of_all	"[[""[IDIOM]"", ""X""], [""certainly"", ""ADV""], [""grateful"", ""ADJ""], [""president"", ""NOUN""], [""leadership"", ""NOUN""], [""advocate"", ""VERB""], [""set_the_table"", ""VERB""], [""reality"", ""NOUN""], [""tax"", ""NOUN""], [""president"", ""NOUN""], [""propose"", ""VERB""], [""plan"", ""NOUN""], [""year"", ""NOUN""], [""ago"", ""ADV""]]"
set_the_table	"[[""[IDIOM]"", ""X""], [""certainly"", ""ADV""], [""grateful"", ""ADJ""], [""president"", ""NOUN""], [""leadership"", ""NOUN""], [""advocate"", ""VERB""], [""[IDIOM]"", ""X""], [""reality"", ""NOUN""], [""tax"", ""NOUN""], [""president"", ""NOUN""], [""propose"", ""VERB""], [""plan"", ""NOUN""], [""year"", ""NOUN""], [""ago"", ""ADV""]]"

"""
import json
from idiom2collocations.corpora import Coca, Opensub
from idiom2collocations.paths import (
    COCA_SPOK_TRAIN_NDJSON,
    OPENSUB_TRAIN_NDJSON,
    IDIOM2SENT_TSV
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
    cock_spok = Coca(COCA_SPOK_TRAIN_NDJSON)
    opensub = Opensub(OPENSUB_TRAIN_NDJSON)
    sentences = chain(cock_spok, opensub)  # chain the generators.

    # I want to check how long it would take just iterating over all of the sentences
    start = time.process_time()
    total = 0
    # --- start writing --- #
    with open(IDIOM2SENT_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idx_sent, sent in enumerate(sentences):
            total += 1
            if len(sent) < 2:  # should be more than 2.
                continue
            for idx_token, token in enumerate(sent):
                if token in idioms:
                    idiom = token
                    sent = sent[:idx_token] + ["[IDIOM]"] + sent[idx_token + 1:]  # just omit it.
                    to_write = [idiom, json.dumps(sent)]
                    tsv_writer.writerow(to_write)
                    # make sure to insert the idiom again
                    sent[idx_token] = idiom

    end = time.process_time()
    print("It took {} minutes to iterate over {} sents".format(str((end - start) / 60), str(total)))


if __name__ == '__main__':
    main()
