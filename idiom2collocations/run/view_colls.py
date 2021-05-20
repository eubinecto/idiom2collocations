from functional.pipeline import Sequence
from idiom2collocations.loaders import load_idiom2colls
import argparse

def print_colls(idiom_given: str, colls: Sequence):
    for idiom, verb_colls, noun_colls, adj_colls, adv_colls in colls:
        if idiom == idiom_given:
            for lemma, score in verb_colls[:6]:
                print("{}({:.2f}),".format(lemma, score), end=" ")
            print("")
            for lemma, score in noun_colls[:6]:
                print("{}({:.2f}),".format(lemma, score), end=" ")
            print("")
            for lemma, score in adj_colls[:6]:
                print("{}({:.2f}),".format(lemma, score), end=" ")
            print("")
            for lemma, score in adv_colls[:6]:
                print("{}({:.2f}),".format(lemma, score), end=" ")
            print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idiom", type=str,
                        default="best_of_both_worlds")
    args = parser.parse_args()
    idiom_given: str  = args.idiom

    idiom2colls_tf = load_idiom2colls('tf')
    idiom2colls_tfidf = load_idiom2colls('tfidf')
    idiom2colls_pmi = load_idiom2colls('pmi')

    print("---tf---")
    print_colls(idiom_given, idiom2colls_tf)
    print("---tfidf---")
    print_colls(idiom_given, idiom2colls_tfidf)
    print("---pmi---")
    print_colls(idiom_given, idiom2colls_pmi)


if __name__ == '__main__':
    main()
