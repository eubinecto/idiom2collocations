import csv
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV, RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_MODEL
import logging
from sys import stdout
from utils import load_idiom2contexts_flattened
logging.basicConfig(stream=stdout, level=logging.DEBUG)


def main():
    # a generator.
    idiom2contexts = load_idiom2contexts_flattened()
    idioms = [idiom for idiom, _ in idiom2contexts]
    docs = [context for _, context in idiom2contexts]
    dct = Dictionary(docs)
    # this is the bows.
    corpus = [
        dct.doc2bow(doc, allow_update=True)
        for doc in docs
    ]
    # fit the model
    tfidf_model = TfidfModel(corpus, smartirs='ntc')
    tfidf_model.save(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_MODEL)  # saving the model for reproducibility.
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for doc, idiom in zip(tfidf_model[corpus], idioms):
            topics_sorted = sorted([(dct[idx], freq) for idx, freq in doc],
                                   key=lambda x: x[1], reverse=True)
            to_write = [idiom, str(topics_sorted[:10])]  # Just write the top 10
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
