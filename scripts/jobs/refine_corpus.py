import json
from typing import List

from pyspark import SparkContext
from merge_idioms.builders import MIPBuilder
from spacy import Language


def load_mip() -> Language:
    mip_builder = MIPBuilder()
    mip_builder.construct()
    print("############## mip loaded ###############")
    return mip_builder.mip


# build mip - expensive
mip = load_mip()


def refine(sent: str) -> List[str]:
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
            # lemmatize
            token.lemma_
            # tokenize
            for token in mip(sent.lower())
            # cleanse
            if not token.is_punct
            if not token.like_num
            if not token.is_stop
        ]
    except ValueError as ve:
        return ["ERROR:sent={},err={}".format(sent, str(ve))]
    else:
        return lemmas


def main():

    sc = SparkContext(appName="refine_corpus")
    # with open(OPENSUB_CORPUS_TXT_PATH, 'r') as fh:
    #     # read all the corpus onto memory
    #     opensub_txt = fh.read()
    # this is..probably an RDD object
    # yup. it is RDD
    # well, if you want to parallelize.. you have to..?
    # this is not going to run in parallel... but better than having to log everything.
    # this is using map reduce.. but won't be executed in parallel.
    # putting
    sc.textFile("gs://dataproc-staging-asia-east1-65124815385-qqgn8yda/data/OpenSubtitles.en-pt_br_en.txt")\
      .map(refine)\
      .map(lambda x: json.dumps(x))\
      .saveAsTextFile("gs://dataproc-staging-asia-east1-65124815385-qqgn8yda/data/OpenSubtitles.en-pt_br_en_refined")


if __name__ == '__main__':
    main()
