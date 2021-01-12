# first, you have to zip the virtualenv
#zip -r idiom2topicsenv.zip idiom2topicsenv
gcloud dataproc jobs submit pyspark \
  ./jobs/word_count.py \
  --cluster=cluster-idiom2topics\
  --region=asia-east1