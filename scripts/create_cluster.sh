REGION=asia-east1
CLUSTER_NAME=cluster-idioms2topics
gcloud dataproc clusters create ${CLUSTER_NAME} \
    --region ${REGION} \
    --image-version 1.5 \
    --metadata 'PIP_PACKAGES=merge-idioms==0.0.4' \
    --initialization-actions gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh,gs://dataproc-staging-asia-east1-65124815385-qqgn8yda/download_en_core_model.sh