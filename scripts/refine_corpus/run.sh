# first, split the corpus
python3 ./split_corpus.py
# then we refine the splits with multiprocessing
python3 ./refine_splits.py
# merge them to a single text
python3 ./merge_splits.py
