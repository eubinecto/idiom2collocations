
from idiom2collocations.loaders import load_idiom2colls


def main():
    idiom2colls_tf = dict(load_idiom2colls(mode='tf').map(lambda x: (x[0], x[1:])))
    idiom2colls_tfidf = dict(load_idiom2colls(mode='tfidf').map(lambda x: (x[0], x[1:])))
    idiom2colls_pmi = dict(load_idiom2colls(mode='pmi').map(lambda x: (x[0], x[1:])))
    idioms = [
        idiom
        for idiom, _ in idiom2colls_tf.items()
    ]
    for idiom in idioms:
        print("### {} ###".format(idiom))
        tf_colls = idiom2colls_tf[idiom]
        tfidf_colls = idiom2colls_tfidf[idiom]
        pmi_colls = idiom2colls_pmi[idiom]
        print("---VERB---")
        print("tf", tf_colls[0][:10])
        print("tfidf", tfidf_colls[0][:10])
        print("pmi", pmi_colls[0][:10])
        print("---NOUN---")
        print("tf", tf_colls[1][:10])
        print("tfidf", tfidf_colls[1][:10])
        print("pmi", pmi_colls[1][:10])
        print("---ADJ---")
        print("tf", tf_colls[2][:10])
        print("tfidf", tfidf_colls[2][:10])
        print("pmi", pmi_colls[2][:10])
        print("---ADV---")
        print("tf", tf_colls[3][:10])
        print("tfidf", tfidf_colls[3][:10])
        print("pmi", pmi_colls[3][:10])


if __name__ == '__main__':
    main()
