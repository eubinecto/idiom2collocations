"""
I need this so that I don't have to build dictionary every single time I load them.
"""
from gensim.corpora import Dictionary
from utils import load_idiom2context
from config import RESULTS_SAMPLE_DICT


def main():
    idiom2context = load_idiom2context()
    docs = [
        context
        for _, context in idiom2context
    ]
    dct = Dictionary(documents=docs)
    dct.save(RESULTS_SAMPLE_DICT)


if __name__ == '__main__':
    main()
