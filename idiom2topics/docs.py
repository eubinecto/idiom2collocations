"""
elasticsearch docs.
"""
# make two indices for pairs here.
# one for pairs_idx, the other for pairs_sample_idx. (the latter is for agile experiment)

from elasticsearch_dsl import Document
from elasticsearch_dsl import Keyword


# this is the one to experiment with in development
# tokens..?
class RefinedSample(Document):
    prev_id = Keyword(multi=False, required=False)
    resp_tokens = Keyword(multi=False, required=True)

    class Index:
        name = "refined_sample_idx"


# this is the one to use with in production
class Refined(Document):
    prev_id = Keyword(multi=False, required=False)
    resp_tokens = Keyword(multi=False, required=True)

    class Index:
        name = "refined_idx"

