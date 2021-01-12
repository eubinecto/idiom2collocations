export GOOGLE_APPLICATION_CREDENTIALS=../data/credentials.json
python3 ./jobs/wordcount_minimal.py \
 --input gs://idiom2topics/OpenSubtitles.en-pt_br_en.txt
