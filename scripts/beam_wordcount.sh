export GOOGLE_APPLICATION_CREDENTIALS=../data/idiom2topics-3894f28ab488.json
python3 ./jobs/wordcount_minimal.py \
 --input gs://idiom2topics/OpenSubtitles.en-pt_br_en.txt
