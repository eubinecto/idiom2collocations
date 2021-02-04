import csv

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from loaders import Idiom2ContextLoader
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2CONTEXT_TSV, RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV
import logging
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.DEBUG)


def main():
    # a generator.
    idiom2contexts = list(Idiom2ContextLoader(RESULTS_SAMPLE_IDIOM2CONTEXT_TSV).load())
    idioms = [
        idiom
        for idiom, _ in idiom2contexts
    ]
    docs = [
        context
        for _, context in idiom2contexts
    ]
    dct = Dictionary(docs)
    corpus = [
        dct.doc2bow(doc, allow_update=True)
        for doc in docs
    ]
    # fit the model
    tfidf_model = TfidfModel(corpus, smartirs='ntc')

    with open(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for doc, idiom in zip(tfidf_model[corpus], idioms):
            topics_sorted = sorted([(dct[idx], freq) for idx, freq in doc],
                                   key=lambda x: x[1], reverse=True)
            to_write = [idiom, str(topics_sorted)]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
