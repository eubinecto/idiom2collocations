from idiom2collocations.loaders import load_idiom2clusters


def view_clusters(idiom_key: str, n: int = 10):
    idiom2clusters = {
        idiom: (a, b, c, d, e, f)
        for idiom, a, b, c, d, e, f in load_idiom2clusters()
    }
    clusters = idiom2clusters[idiom_key]
    print("### {} ###".format(idiom_key))
    print("x_idiom")
    print(clusters[0][:n])
    print("idiom_x")
    print(clusters[1][:n])
    print("xx_idiom")
    print(clusters[2][:n])
    print("idiom_xx")
    print(clusters[3][:n])
    print("xxx_idiom")
    print(clusters[4][:n])
    print("idiom_xxx")
    print(clusters[5][:n])


def view_collocations(idiom_key: str, n: int = 10):
    pass
