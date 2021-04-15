import csv
import json
from idiom2collocations.loaders import load_idiom2sent
from idiom2collocations.models import ClusterModel
from idiom2collocations.paths import (
    IDIOM2CLUSTERS_TSV
)
import logging
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.INFO)


def main():
    # --- load the data --- #
    idiom2sent = load_idiom2sent()
    # --- instantiate the model --- #
    cluster_model = ClusterModel(idiom2sent)
    # --- fit the cluster_model --- #
    cluster_model.fit()
    # --- save the collocations --- #
    with open(IDIOM2CLUSTERS_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for idiom in cluster_model.idioms:
            x_idiom = cluster_model.x_idiom[idiom]
            idiom_x = cluster_model.idiom_x[idiom]
            xx_idiom = cluster_model.xx_idiom[idiom]
            idiom_xx = cluster_model.idiom_xx[idiom]
            xxx_idiom = cluster_model.xxx_idiom[idiom]
            idiom_xxx = cluster_model.idiom_xxx[idiom]
            to_write = [
                idiom,
                json.dumps(x_idiom),
                json.dumps(idiom_x),
                json.dumps(xx_idiom),
                json.dumps(idiom_xx),
                json.dumps(xxx_idiom),
                json.dumps(idiom_xxx)
            ]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
