"""
idiom, verb_context, noun_context, adj_context, adv_context
to be used for collocation extraction.
serialise a count dictionary, rather than the full list. That makes more sense.
We've got this from within the window.
"""
import csv
import json
from multiprocessing import Pool
from typing import Tuple, List
from nltk import ngrams
from tqdm import tqdm
from idiom2collocations.loaders import load_idiom2lemma2pos
from idiom2collocations.paths import IDIOM2BOWS_TSV

WINDOW: int = 3  # three is appropriate.
WORKERS: int = 4


def process_idiom2lemma2pos(pair: Tuple[str, List[Tuple[str, str]]]):
    idiom = pair[0]
    lemma2pos = pair[1]
    verb_bow = dict()
    noun_bow = dict()
    adj_bow = dict()
    adv_bow = dict()
    # we are only getting those that include the idiom.
    for ngram in ngrams(lemma2pos, n=WINDOW):
        if "[IDIOM]" in [lemma for lemma, _ in ngram]:
            for lemma, pos in ngram:
                if lemma != "[IDIOM]":
                    if pos == "VERB":
                        verb_bow[lemma] = verb_bow.get(lemma, 1) + 1
                    elif pos == "NOUN":
                        noun_bow[lemma] = noun_bow.get(lemma, 1) + 1
                    elif pos == "ADJ":
                        adj_bow[lemma] = adj_bow.get(lemma, 1) + 1
                    elif pos == "ADV":
                        adv_bow[lemma] = adv_bow.get(lemma, 1) + 1
    return idiom, verb_bow, noun_bow, adj_bow, adv_bow


def main():
    idiom2lemma2pos = load_idiom2lemma2pos()
    total = len(list(idiom2lemma2pos))
    with Pool(4) as p:
        # In order to show progress bar,
        # https://github.com/tqdm/tqdm/issues/484#issuecomment-461998250
        idiom2bows = list(tqdm(p.imap(process_idiom2lemma2pos, idiom2lemma2pos), total=total))

    with open(IDIOM2BOWS_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom, a, b, c, d in idiom2bows:
            to_write = [idiom, json.dumps(a), json.dumps(b), json.dumps(c), json.dumps(d)]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
