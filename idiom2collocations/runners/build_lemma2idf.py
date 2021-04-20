from idiom2collocations.loaders import load_idiom2bows
from idiom2collocations.models import TFIDFCollModel
from idiom2collocations.paths import LEMMA2IDF_TSV
import csv


def main():
    # instantiate, and fit the model
    idiom2bows = load_idiom2bows()
    coll_model = TFIDFCollModel(idiom2bows)
    coll_model.fit()

    with open(LEMMA2IDF_TSV, 'r') as fh:
        tsv_writer = csv.writer(fh, delimiter='\t')
        for token, token_idx in coll_model.dct.token2id.items():
            verb_idf = coll_model.verb_tfidf.idfs[token_idx]
            noun_idf = coll_model.noun_tfidf.idfs[token_idx]
            adj_idf = coll_model.adj_tfidf.idfs[token_idx]
            adv_idf = coll_model.adv_tfidf.idfs[token_idx]
            to_write = [token, verb_idf, noun_idf, adj_idf, adv_idf]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
