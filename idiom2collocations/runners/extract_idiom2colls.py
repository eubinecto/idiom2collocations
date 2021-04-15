import argparse
import csv
import json
from idiom2collocations.loaders import load_idiom2bows
from idiom2collocations.models import TFCollModel, PMICollModel, TFIDFCollModel
from idiom2collocations.paths import (
    IDIOM2COLLS_TF_TSV,
    IDIOM2COLLS_TFIDF_TSV,
    IDIOM2COLLS_PMI_TSV
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",
                        type=str,
                        default="tf")

    args = parser.parse_args()
    model_type: str = args.model_type
    idiom2bows = load_idiom2bows()
    # --- instantiate the model and tsv path --- #
    if model_type == "tf":
        model = TFCollModel(idiom2bows)
        tsv_path = IDIOM2COLLS_TF_TSV
    elif model_type == "tfidf":
        model = TFIDFCollModel(idiom2bows)
        tsv_path = IDIOM2COLLS_TFIDF_TSV
    elif model_type == "pmi":
        # what other models do we have..?
        model = PMICollModel(idiom2bows)
        tsv_path = IDIOM2COLLS_PMI_TSV
    else:
        raise ValueError("Invalid model_type: " + model_type)

    # extract the collocations
    model.fit()

    # then, save the collocations as tsv
    with open(tsv_path, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom in model.idioms:
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
