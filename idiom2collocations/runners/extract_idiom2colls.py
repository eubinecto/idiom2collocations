import argparse
from idiom2collocations.loaders import load_idiom2contexts
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

    idiom2contexts = load_idiom2contexts()

    # --- instantiate the model and tsv path --- #
    if model_type == "tf":
        model = TFCollModel(idiom2contexts)
        tsv_path = IDIOM2COLLS_TF_TSV
    elif model_type == "tfidf":
        model = TFIDFCollModel(idiom2contexts)
        tsv_path = IDIOM2COLLS_TFIDF_TSV
    elif model_type == "pmi":
        # what other models do we have..?
        model = PMICollModel(idiom2contexts)
        tsv_path = IDIOM2COLLS_PMI_TSV
    else:
        raise ValueError("Invalid model_type: " + model_type)

    model.fit()

    # then, save the collocations as tsv
    # TODO: save.


if __name__ == '__main__':
    main()