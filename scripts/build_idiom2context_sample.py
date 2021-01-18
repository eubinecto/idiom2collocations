import json
from typing import List
from merge_idioms.loaders import TargetIdiomsLoader
from idiom2topics.builders import SearchBuilder
from idiom2topics.docs import RefinedSample
from idiom2topics.config import (
    RESULTS_SAMPLE_IDIOM2CONTEXT_NDJSON,
)
CONTEXT_LENGTH = 1  # this is a big---- problem. I suppose. Should be a port of a discussion.
SRCH_FIELD = "resp_tokens.keyword"


def collect_context(idiom: str, source: dict) -> List[str]:
    global CONTEXT_LENGTH
    context: List[str] = list()
    resp_tokens: List[str] = source['resp_tokens']
    near_past_idx = resp_tokens.index(idiom)
    context += resp_tokens[:near_past_idx]
    look_up_id = source.get('prev_id', None)
    for _ in range(CONTEXT_LENGTH):
        if look_up_id:
            source = RefinedSample.get(id=look_up_id).to_dict()
            context += source['resp_tokens']
            look_up_id = source.get('prev_id', None)
    else:
        return context


def main():
    global CONTEXT_LENGTH, SRCH_FIELD

    with open(RESULTS_SAMPLE_IDIOM2CONTEXT_NDJSON, 'w') as fh:
        for idiom in TargetIdiomsLoader().load():
            s_builder = SearchBuilder()
            s_builder.construct(text=idiom, field=SRCH_FIELD, size=10000)
            r = s_builder.search.execute()
            r_dict = r.to_dict()
            hits_list = r_dict['hits'].get('hits', None)
            if hits_list:
                sources = (hit['_source'] for hit in hits_list)
                for source in sources:
                    context = collect_context(idiom, source)
                    to_write = [idiom, context]
                    # write in the format of ndjson
                    fh.write(json.dumps(to_write) + "\n")
            else:
                to_write = [idiom, list()]
                fh.write(json.dumps(to_write) + "\n")


if __name__ == '__main__':
    main()
