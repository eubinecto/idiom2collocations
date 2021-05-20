"""
for having a look at the distributions of the idioms.
"""
from idiom2collocations.loaders import load_idiom2sent
from idiom2collocations.paths import IDIOM2FREQ_TSV
import csv


def main():
    idiom2sent = load_idiom2sent()
    idiom2freq = dict()
    for idiom, _ in idiom2sent:
        idiom2freq[idiom] = idiom2freq.get(idiom, 0) + 1

    with open(IDIOM2FREQ_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom, freq in sorted(idiom2freq.items(), key=lambda x: x[1], reverse=False):
            to_write = [idiom, freq]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()

