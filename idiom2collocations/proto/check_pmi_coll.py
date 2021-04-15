from idiom2collocations.loaders import load_idiom2sent
from idiom2collocations.models import PMICollModel
import logging
from sys import stdout
import time
logging.basicConfig(stream=stdout, level=logging.INFO)


def main():
    idiom2sent = load_idiom2sent()
    pmi_coll_model = PMICollModel(idiom2sent)
    start = time.process_time()
    pmi_coll_model.fit()  # this is taking way too much time, maybe
    end = time.process_time()
    print("time took: {} mins".format(str((end - start)/ 60)))
    for idiom, x_idioms in pmi_coll_model.x_idiom_colls:
        print("### {} ###".format(idiom))
        print(x_idioms)


if __name__ == '__main__':
    main()
