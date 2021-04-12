from utils import load_idiom2contexts_flattened, load_dictionary
from config import RESULTS_SAMPLE_IDIOM2TOPICS_TF_NDJSON
import json


def main():
    # a generator.
    idiom2contexts = load_idiom2contexts_flattened()
    dct = load_dictionary()
    idioms = [idiom for idiom, _ in idiom2contexts]
    docs = [context for _, context in idiom2contexts]
    # this is the bows.
    corpus = [
        dct.doc2bow(doc, allow_update=True)
        for doc in docs
    ]

    with open(RESULTS_SAMPLE_IDIOM2TOPICS_TF_NDJSON, 'w') as fh:
        for idiom, bow in zip(idioms, corpus):
            bow_decoded = [
                (dct[tok_id], tf)
                for tok_id, tf in bow
            ]
            bow_sorted = sorted(bow_decoded, key=lambda x: x[1], reverse=True)
            entry = {
                'idiom': idiom,
                'tfs': bow_sorted
            }
            fh.write(json.dumps(entry) + "\n")


if __name__ == '__main__':
    main()
