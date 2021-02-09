from typing import List, Optional
from elasticsearch_dsl import Search
from config import ES_CLIENT
from docs import RefinedSample
from typing import Callable


class Builder:
    def construct(self, *args):
        for step in self.steps():
            step()

    def steps(self) -> List[Callable]:
        raise NotImplementedError


# first, the interface
class SearchBuilder(Builder):
    # common attribute
    # global constants
    PRE_TAG = "<strong>"
    POST_TAG = "</strong>"
    NUM_FRAGMENTS: int = 2
    # parameters for n-gram search.

    def __init__(self):
        # the body of query.
        self.body: Optional[dict] = None
        # must-include parameters.
        self.text: Optional[str] = None
        self.from_: Optional[int] = None
        self.size: Optional[int] = None
        self.field: Optional[str] = None
        # the one to build
        self.search: Optional[Search] = None

    def construct(self, text: str, field: str, from_: int = 0, size: int = 10):
        self.text: str = text  # the text to search on the index.
        self.body = dict()
        self.from_: int = from_
        self.size: int = size
        self.field: str = field
        super(SearchBuilder, self).construct()

    def steps(self) -> List[Callable]:
        return [
            self.prepare,
            self.build_query,
            self.build_highlight,
            self.build_search
        ]

    # this should be the first step.
    def prepare(self):
        self.body.update(
            {
                "query": dict(),
                # for pagination, we need these two properties.
                "from": self.from_,
                "size": self.size
            }
        )  # update body.

    def build_query(self):
        self.body['query'].update(
            {
                "term": {
                    self.field: {
                        "value": self.text
                    }
                }
            }
        )

    def build_highlight(self):
        # update highlight.
        self.body.update(
            {
                "highlight": {
                    "pre_tags": [self.PRE_TAG],
                    "post_tags": [self.POST_TAG],
                    "fields": {
                        self.field: {
                            "number_of_fragments": self.NUM_FRAGMENTS,
                        }
                    }
                }
            }
        )

    def build_search(self):
        search = Search(using=ES_CLIENT, index=RefinedSample.Index.name)
        search.update_from_dict(self.body)
        self.search = search
