import csv
import json
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from loaders import Idiom2ContextLoader
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2CONTEXT_TSV, RESULTS_SAMPLE_IDIOM2TOPICS_LDA_TSV
import logging
from sys import stdout
from functional import seq
import numpy as np  # you may need this for getting the average..
logging.basicConfig(stream=stdout, level=logging.DEBUG)


def main():
    # a generator. - list of tuples.
    idiom2context = Idiom2ContextLoader(RESULTS_SAMPLE_IDIOM2CONTEXT_TSV).load()
    # turning them into a list, as we need to iterate over them more than once.
    idiom2docs = seq(idiom2context).group_by_key().to_list()  # idiom, context -> group by idiom.
    # load the docs.
    # question - how do I stream a document.. for building a topic modeling?
    docs = [
        doc
        for _, docs in idiom2docs
        for doc in docs
    ]  # flatten out all the docs
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
                             passes=4)
    # have a guess, what are those topics? - how do I see them...?
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom, docs in idiom2docs:
            bows = [dct.doc2bow(doc) for doc in docs]
            topic_vectors = [
                [prob for _, prob in lda_model.get_document_topics(bow, minimum_probability=0)]
                for bow in bows
            ]
            # should be able to do this...
            topic_mat = np.array(topic_vectors)
            # TODO: use gaussian distribution over each column, instead of just using the mean.
            # then, you could use the probability of getting a scalar from that distribution (the output of pdf)
            # as how close each example is.
            idiom_topic_vector = topic_mat.mean(axis=0)  # mean over each column
            to_write = [idiom, json.dumps(idiom_topic_vector.tolist())]
            tsv_writer.writerow(to_write)   # write them.. this is essentially the representation of them.


if __name__ == '__main__':
    main()
