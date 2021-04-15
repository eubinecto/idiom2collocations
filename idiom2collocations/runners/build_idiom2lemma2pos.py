"""
tag idiom2sent with part-of-speech.
all right. we are going to have some experiment with a small dataset today.
just do it.
"""
import re
import csv
import json
from typing import Tuple, List
from spacy import load, Language
from multiprocessing.pool import Pool
from tqdm import tqdm
from idiom2collocations.loaders import load_idiom2sent
from idiom2collocations.paths import NLP_MODEL, IDIOM2LEMMA2POS_TSV

nlp: Language = load(NLP_MODEL)


def process_idiom2sent(pair: Tuple[str, List[str]]) -> Tuple[str, List[Tuple[str, str]]]:
    global nlp
    nlp.tokenizer.add_special_case("[IDIOM]", [{"ORTH": "[IDIOM]"}])  # not supposed to tokenise this.
    idiom = pair[0]
    sent = pair[1]
    untokenized = " ".join(sent)
    lemma2pos = [
        (token.text if token.text == "[IDIOM]" else token.lemma_, token.pos_)
        for token in nlp(untokenized)
        # do some cleanup here.
        if token.pos_ != "PROPN"  # we don't need proper nouns.
        if not token.is_stop
        if not token.like_num
        if not token.is_punct
        if not re.match(r'^[A-Z\'@!-\(\)]+$', token.text)
    ]
    return idiom, lemma2pos


def main():
    idiom2sent = load_idiom2sent()  # test.
    total = len(list(idiom2sent))
    # --- execute the process with parallelism --- #
    with Pool(4) as p:
        # In order to show progress bar,
        # https://github.com/tqdm/tqdm/issues/484#issuecomment-461998250
        idiom2lemma2pos = list(tqdm(p.imap(process_idiom2sent, idiom2sent), total=total))

    with open(IDIOM2LEMMA2POS_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom, lemma2pos in idiom2lemma2pos:
            to_write = [idiom, json.dumps(lemma2pos)]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
