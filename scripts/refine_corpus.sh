# for starting the job
spark-submit ../idiom2topics/spark_jobs/refine_corpus.py
# for merging the output
#hadoop fs -getmerge "$OPENSUB_CORPUS_REFINED_DIR"  "$OPENSUB_CORPUS_REFINED_TXT_PATH"