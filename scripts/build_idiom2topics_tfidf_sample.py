import json
from typing import List, cast
from pyspark import SparkContext
from pyspark.ml.linalg import Vector
from pyspark.sql import SparkSession
from pyspark.sql import functions as sql_F
from pyspark.sql.types import StructType, StringType, ArrayType
from idiom2topics.config import RESULTS_SAMPLE_IDIOM2CONTEXT_NDJSON, \
    RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_SPLITS_DIR, \
    RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF, IDFModel
import os


# pretty good example tutorial: https://sparkbyexamples.com/pyspark/pyspark-read-csv-file-into-dataframe/


# using currying: https://stackoverflow.com/a/35375306
def to_topics(weight_vector: Vector, vocab_list: List[str]) -> str:
    """
    joins the idx_vector with tf_vector.
    :param weight_vector: could be tf_vector, or tf*idf vector
    :return:
    """
    tf_list: List[float] = cast(list, weight_vector.toArray().tolist())
    topics = [
        (word, tf)
        for word, tf in zip(vocab_list, tf_list)
        if tf > 0.0  # only interested in those greater than 0.
    ]
    topics = sorted(topics, key=lambda x: x[1], reverse=True)
    return str(topics)


def to_topics_udf(vocab_list: List[str]):
    return sql_F.udf(lambda x: to_topics(x, vocab_list), StringType())


def main():
    sc = SparkContext(master="local[*]", appName="build_idiom2topics_tf_sample")
    spark = SparkSession(sc)  # in order to use Dataframe API, you need spark session.

    schema = StructType() \
        .add("idiom", StringType(), False) \
        .add("context", ArrayType(elementType=StringType()), True)

    # note: right after df.rdd, the return value is row
    df = spark.read.text(RESULTS_SAMPLE_IDIOM2CONTEXT_NDJSON) \
        .rdd.map(lambda row: json.loads(row[0])) \
        .map(lambda x: (x[0], x[1])) \
        .toDF(schema)

    # merge all the contexts
    merged_df = df.groupby("idiom") \
        .agg(sql_F.collect_list("context").alias("context_all")) \
        .select("idiom", sql_F.flatten("context_all").alias("context_merged"))

    # add a tf_vector column to merged_df
    cv = CountVectorizer()
    cv_model: CountVectorizerModel = cv.setInputCol("context_merged") \
        .setOutputCol("tf_vector") \
        .fit(merged_df)
    tf_df = cv_model.transform(merged_df)

    # scale tf's by multiplying idf's to them.
    idf = IDF()
    idf_model: IDFModel = idf.setInputCol("tf_vector") \
        .setOutputCol("tfidf_vector") \
        .fit(tf_df)
    tfidf_df = idf_model.transform(tf_df)

    idiom2topics_tfidf_df = tfidf_df.select("idiom",
                                         # currying.
                                         to_topics_udf(cv_model.vocabulary)(tfidf_df.tfidf_vector)
                                         .alias("topics")
                                         )
    idiom2topics_tfidf_df.write.csv(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_SPLITS_DIR, header=False, sep="\t")
    # merge the results
    os.system("hadoop fs -getmerge {} {}".format(RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_SPLITS_DIR,
                                                 RESULTS_SAMPLE_IDIOM2TOPICS_TFIDF_TSV))


if __name__ == '__main__':
    main()
