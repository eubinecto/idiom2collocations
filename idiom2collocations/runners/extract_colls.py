import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        type=str,
                        default="tf")

    args = parser.parse_args()
    method: str = args.method

    # --- choose the method --- #
    if method == "tf":
        pass
    elif method == "tfidf":
        pass
    elif method == "pmi":
        pass
    else:
        raise ValueError("Invalid method: " + method)


if __name__ == '__main__':
    main()