from elasticsearch_dsl import Document

from config import ES_CLIENT
from docs import RefinedSample


def main():
    # delete and init.
    RefinedSample.init(using=ES_CLIENT)
    # do this later.
    # Refined.init(using=ES_CLIENT)


if __name__ == '__main__':
    main()
