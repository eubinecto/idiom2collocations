from merge_idioms.builders import MIPBuilder
from pyspark import SparkContext
from spacy import Language
from idiom2topics.config import OPENSUB_CORPUS_TXT_PATH, OPENSUB_CORPUS_REFINED_DIR
import json


# TODO: in merge_idioms, having load_mip() would be nice
def load_mip() -> Language:
    mip_builder = MIPBuilder()
    mip_builder.construct()
    print("############## mip loaded ###############")
    return mip_builder.mip


# global access point
# hopefully.. this can be registered alongside udf?
mip = load_mip()


# functions for preprocessing, in functional paradigm.
def refine(sent: str) -> str:
    """
    1. tokenise
    2. cleanse
    3. lemmatize
    :param sent:
    :return:
    """
    global mip
    try:
        lemmas = [
            token.lemma_
            for token in mip(sent.lower())
            if not token.is_punct
            if not token.like_num
            if not token.is_stop
        ]
    except ValueError:
        print("##########")
        print(sent)
        print("##########")
        return json.dumps(["ERROR:sent=".format(sent)])  # returns an empty list
    else:
        return json.dumps(lemmas)
    # return [
    #     token
    #     for token in mip(sent)
    # ]


def main():
    # spark job
    sc = SparkContext("local[*]", "refine_corpus")
    # read the text file.
    # map each line to list of refined tokens
    sc.textFile(OPENSUB_CORPUS_TXT_PATH)\
      .map(refine)\
      .saveAsTextFile(OPENSUB_CORPUS_REFINED_DIR)


if __name__ == '__main__':
    main()
