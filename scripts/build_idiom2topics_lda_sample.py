import csv
from pprint import PrettyPrinter
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from loaders import Idiom2ContextLoader
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2CONTEXT_TSV
import logging
from sys import stdout
from functional import seq
import numpy as np  # you may need this for getting the average..
logging.basicConfig(stream=stdout, level=logging.DEBUG)


def main():
    # a generator. - list of tuples.
    idiom2context = Idiom2ContextLoader(RESULTS_SAMPLE_IDIOM2CONTEXT_TSV).load()
    idiom2docs = seq(idiom2context).group_by_key().to_list()  # idiom, context -> group by idiom.
    # load the docs.
    # question - how do I stream a document.. for building a topic modeling?
    docs = [
        doc
        for _, docs in idiom2docs
        for doc in docs
    ]
    # need all of them to be loaded to build a dictionary.
    dct = Dictionary(documents=docs)
    # build bag-of-words representations.
    bows = [
        dct.doc2bow(doc, allow_update=True)
        for doc in docs
    ]

    # TODO: diff between iterations & passes? I see "passes" = "epochs", but..
    # so.. it was a matter of tuning this hyper parameter.
    # I certainly don't want to have a model for each idiom.
    # you must pass the.. encoded version of bows. (they must be integers)
    lda_model = LdaMulticore(bows,
                             workers=4,
                             # for debugging and topic printing, we need to give it this.
                             id2word=dct,
                             num_topics=100,
                             passes=10)
    # have a guess, what are those topics? - how do I see them...?
    for topic_id, idx2prob in lda_model.show_topics(num_topics=100, num_words=10, formatted=True):
        print(topic_id, "--->", idx2prob)


if __name__ == '__main__':
    main()
