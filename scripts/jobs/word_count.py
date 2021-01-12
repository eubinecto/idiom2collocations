from pyspark import SparkContext
# for merging the output, use hadoop
# https://stackoverflow.com/a/5737935
# hadoop fs -getmerge data/opensub_corpus/word_count  data/opensub_corpus/word_count.txt


def main():
    # as many as logical core:
    # https://spark.apache.org/docs/latest/submitting-applications.html#master-urls
    # how do I create a spark session?
    sc = SparkContext(appName="PySpark Word Count Example")
    # with open(OPENSUB_CORPUS_TXT_PATH, 'r') as fh:
    #     # read all the corpus onto memory
    #     opensub_txt = fh.read()
    # this is..probably an RDD object
    # yup. it is RDD
    # well, if you want to parallelize.. you have to..?
    # this is not going to run in parallel... but better than having to log everything.
    # this is using map reduce.. but won't be executed in parallel.
    # putting

    wordcnts_rdd = sc.textFile("gs://dataproc-staging-asia-east1-65124815385-qqgn8yda/data/OpenSubtitles.en-pt_br_en.txt")\
                     .flatMap(lambda line: line.split(" "))\
                     .map(lambda word: (word, 1))\
                     .reduceByKey(lambda cnt_1, cnt_2: cnt_1 + cnt_2)\
                     .sortBy(lambda x: x[1], ascending=False)
    # if you want to save rdd, use this
    wordcnts_rdd.saveAsTextFile("gs://dataproc-staging-asia-east1-65124815385-qqgn8yda/data/OpenSubtitles.en-pt_br_en_word_count")


if __name__ == '__main__':
    main()
