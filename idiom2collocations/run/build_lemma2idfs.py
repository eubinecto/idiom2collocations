from idiom2collocations.loaders import load_idiom2bows
from idiom2collocations.models import TFIDFCollModel
from idiom2collocations.paths import LEMMA2IDFS_TSV
import csv


def main():
    # instantiate, and fit the model
    idiom2bows = load_idiom2bows()
    coll_model = TFIDFCollModel(idiom2bows)
    coll_model.fit()

    with open(LEMMA2IDFS_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter='\t')
        for token, token_idx in coll_model.dct.token2id.items():
            verb_idf = coll_model.verb_tfidf.idfs.get(token_idx, 1)
            noun_idf = coll_model.noun_tfidf.idfs.get(token_idx, 1)
            adj_idf = coll_model.adj_tfidf.idfs.get(token_idx, 1)
            adv_idf = coll_model.adv_tfidf.idfs.get(token_idx, 1)
            to_write = [token, verb_idf, noun_idf, adj_idf, adv_idf]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
