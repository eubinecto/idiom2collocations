import csv
from pprint import PrettyPrinter

from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from loaders import Idiom2ContextLoader
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2CONTEXT_TSV, RESULTS_SAMPLE_IDIOM2TOPICS_LDA_TSV
import logging
from sys import stdout
from functional import seq
import numpy as np
logging.basicConfig(stream=stdout, level=logging.DEBUG)


def main():
    # a generator. - list of tuples.
    idiom2context = Idiom2ContextLoader(RESULTS_SAMPLE_IDIOM2CONTEXT_TSV).load()
    idiom2docs = seq(idiom2context).group_by_key().to_list()  # idiom, context -> group by idiom.
    # load the docs.
    docs = [
        doc
        for _, docs in idiom2docs
        for doc in docs
    ]
    # need all of them to be loaded to build a dictionary.
    dct = Dictionary(list(docs))
    bows = [
        dct.doc2bow(doc, allow_update=True)
        for doc in docs
    ]
    # TODO: you might want to do.. group by. idiom. get all the bows. - this is probably where you need
    # TODO: diff between iteratiosn & passes? I see "passes" = "epochs", but..
    # so.. it was a matter of tuning this hyper parameter.
    # I certainly don't want to have a model for each idiom.
    lda_model = LdaMulticore(bows, workers=4, num_topics=3)

    # have a guess, what are those topics?
    print(lda_model.show_topics())


if __name__ == '__main__':
    main()
