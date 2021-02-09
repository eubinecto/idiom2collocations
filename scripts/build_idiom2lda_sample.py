import csv
import json
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from os import makedirs
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TSV,\
    RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TRAINED_DIR, \
    RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_MODEL,\
    RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TOPICS_TSV, \
    RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_META_JSON
import logging
from sys import stdout
from functional import seq
import numpy as np  # you may need this for getting the average..
import time

from utils import load_idiom2contexts

logging.basicConfig(stream=stdout, level=logging.DEBUG)


META_DICT: dict = {
    'num_workers': 4,
    'num_topics': 140,
    'num_words': 10,
    'num_passes': 10,
    'num_iterations': 200,  # the higher... the better?
    'group_by': 'avg'
}


def main():
    global META_DICT
    idiom2contexts = load_idiom2contexts()
    # load the docs.
    # question - how do I stream a document.. for building a topic modeling?
    # TODO: just get load the docs directly from tsv. not.. the sequence.
    docs = [
        context
        for _, contexts in idiom2contexts
        for context in contexts
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
    # also...
    start = time.process_time()
    lda_model = LdaMulticore(bows,
                             iterations=META_DICT['num_iterations'],
                             workers=META_DICT['num_workers'],
                             # for debugging and topic printing, we need to give it this.
                             id2word=dct,
                             num_topics=META_DICT['num_topics'],
                             passes=META_DICT['num_passes'])
    META_DICT['train_took'] = time.process_time() - start
    # populate the dirs, then save the model
    makedirs(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TRAINED_DIR)
    lda_model.save(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_MODEL)
    # then, save the topics
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TOPICS_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for topic_id, distribution in lda_model.show_topics(num_topics=META_DICT['num_topics'],
                                                            num_words=META_DICT['num_words'],
                                                            formatted=True):
            to_write = [topic_id, distribution]
            tsv_writer.writerow(to_write)

    # then, save meta.json
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_META_JSON, 'w') as fh:
        fh.write(json.dumps(META_DICT))

    # then, save idiom2topics
    with open(RESULTS_SAMPLE_IDIOM2TOPICS_LDA_ATTEMPT_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom, docs in idiom2docs:
            bows = [dct.doc2bow(doc) for doc in docs]
            topic_vectors = [
                [prob for _, prob in lda_model.get_document_topics(bow, minimum_probability=0)]
                for bow in bows
            ]
            topic_mat = np.array(topic_vectors)
            # TODO: use gaussian distribution over each column, instead of just using the mean.
            # then, you could use the probability of getting a scalar from that distribution (the output of pdf)
            # as how close each example is.
            idiom_topic_vector = topic_mat.mean(axis=0)  # mean over each column
            to_write = [idiom, json.dumps(idiom_topic_vector.tolist())]
            tsv_writer.writerow(to_write)   # write them.. this is essentially the representation of them.

    # then, save all the topics


if __name__ == '__main__':
    main()
