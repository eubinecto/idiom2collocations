import argparse
import csv
import json
from idiom2collocations.loaders import load_idiom2bows, load_idiom2lemma2pos
from idiom2collocations.models import TFCollModel, PMICollModel, TFIDFCollModel
from idiom2collocations.paths import (
    IDIOM2COLLS_TF_TSV,
    IDIOM2COLLS_TFIDF_TSV,
    IDIOM2COLLS_PMI_TSV
)
import logging
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",
                        type=str,
                        default="pmi")
    # to be used for the case of using pmi
    parser.add_argument("--lower_bound",
                        type=int,
                        # greater than this bound.
                        default=3)

    args = parser.parse_args()
    model_type: str = args.model_type
    lower_bound: int = args.lower_bound
    idiom2bows = load_idiom2bows()
    idiom2lemma2pos = load_idiom2lemma2pos()
    # --- instantiate the model and tsv path --- #
    if model_type == "tf":
        model = TFCollModel(idiom2bows)
        tsv_path = IDIOM2COLLS_TF_TSV
    elif model_type == "tfidf":
        model = TFIDFCollModel(idiom2bows)
        tsv_path = IDIOM2COLLS_TFIDF_TSV
    elif model_type == "pmi":
        # you can apply a lower bound, as for pmi.
        model = PMICollModel(idiom2bows, idiom2lemma2pos, lower_bound=lower_bound)
        tsv_path = IDIOM2COLLS_PMI_TSV
    else:
        raise ValueError("Invalid model_type: " + model_type)

    # extract the collocations
    model.fit()

    # then, save the collocations as tsv
    with open(tsv_path, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom in sorted(model.idiom_keys):
            to_write = [
                idiom,
                json.dumps(model.verb_colls[idiom]),
                json.dumps(model.noun_colls[idiom]),
                json.dumps(model.adj_colls[idiom]),
                json.dumps(model.adv_colls[idiom]),
            ]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
