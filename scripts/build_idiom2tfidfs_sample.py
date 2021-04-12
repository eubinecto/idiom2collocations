import json
from os import makedirs
from gensim.models import TfidfModel
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_NDJSON,\
    RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_MODEL,\
    RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR
import logging
from sys import stdout
from utils import load_idiom2contexts_flattened, load_dictionary
logging.basicConfig(stream=stdout, level=logging.DEBUG)


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
    # fit the model
    tfidf_model = TfidfModel(corpus, smartirs='ntc')
    makedirs(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_DIR, exist_ok=True)
    tfidf_model.save(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_MODEL)  # saving the model for reproducibility.
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_NDJSON, 'w') as fh:
        for doc, idiom in zip(tfidf_model[corpus], idioms):
            tfidfs = [
                {'term': dct[idx], 'tfidf': tfidf}
                for idx, tfidf in doc
            ]
            tfidfs_sorted = sorted(tfidfs, key=lambda x: x['tfidf'], reverse=True)
            idiom2topics = {
                'idiom': idiom,
                'tfidfs': tfidfs_sorted
            }
            to_write = json.dumps(idiom2topics) + "\n"
            fh.write(to_write)


if __name__ == '__main__':
    main()
