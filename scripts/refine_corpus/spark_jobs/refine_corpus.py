from typing import List

from pyspark import SparkContext
from idiom2topics.config import OPENSUB_CORPUS_TXT_PATH
from utils import load_mip

mip = load_mip()

def refine(sent: str) -> List[str]:
    """
    1. tokenise
    2. cleanse
    3. lemmatize
    :param sent:
    :return:
    """
    try:
        lemmas = [
            token.lemma_
            for token in MIP(sent.lower())
            if not token.is_punct
            if not token.like_num
            if not token.is_stop
        ]
    except ValueError:
        print("##########")
        print(sent)
        print("##########")
        return ["ERROR:sent=".format(sent)]  # returns an empty list
    else:
        return lemmas
    # return [
    #     token
    #     for token in mip(sent)
    # ]

def main():
    # how do I create a spark session?
    sc = SparkContext("local", "PySpark Word Count Example")
    # with open(OPENSUB_CORPUS_TXT_PATH, 'r') as fh:
    #     # read all the corpus onto memory
    #     opensub_txt = fh.read()
    # this is..probably an RDD object
    # yup. it is RDD
    # well, if you want to parallelize.. you have to..?
    # this is not going to run in parallel... but better than having to log everything.
    # this is using map reduce.. but won't be executed in parallel.
    # putting
    wordcnts_rdd = sc.textFile(OPENSUB_CORPUS_TXT_PATH)\
                    .flatMap(lambda line: line.split(" "))\
                    .map(lambda word: (word, 1))\
                    .reduceByKey(lambda cnt_1, cnt_2: cnt_1 + cnt_2)\
                    .sortBy(lambda x: x[1], ascending=False)
    # if you want to save rdd, use this
    wordcnts_rdd.saveAsTextFile(OPENSUB_CORPUS_WORDCNT_PATH)


if __name__ == '__main__':
    main()