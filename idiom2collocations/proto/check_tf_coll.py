from idiom2collocations.models import TFCollModel
from idiom2collocations.loaders import load_idiom2sent


def main():
    idiom2sent = load_idiom2sent()
    tf_coll_model = TFCollModel(idiom2sent)
    tf_coll_model.fit()

    for idiom, x_idioms in tf_coll_model.x_idiom_colls.items():
        print("### {} ###".format(idiom))
        print(x_idioms)


if __name__ == '__main__':
    main()
