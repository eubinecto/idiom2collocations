import json
from typing import List
from elasticsearch.helpers import bulk
from docs import RefinedSample
from idiom2topics.config import (
    CORPUS_SAMPLE_MERGED_REFINED_NDJSON, ES_CLIENT, BATCH_SIZE
)
# import logging
# from sys import stdout
# logging.basicConfig(stream=stdout, level=logging.INFO)


def index_batch(batch: List[RefinedSample]):
    batch_actions = (refined_sample.to_dict(include_meta=True) for refined_sample in batch)
    bulk(client=ES_CLIENT, actions=batch_actions)
    batch.clear()


def main():
    batch: List[RefinedSample] = list()
    with open(CORPUS_SAMPLE_MERGED_REFINED_NDJSON, 'r') as fh:
        prev_idx = None
        # how do you know.. that next idx won't exist?
        for curr_idx, line in enumerate(fh):
            line = line.replace("%", "").replace("=", "")
            resp_tokens = json.loads(line)
            if resp_tokens:
                refined_sample = RefinedSample(meta={"id": curr_idx},
                                               resp_tokens=resp_tokens,
                                               prev_id=prev_idx)
                batch.append(refined_sample)
                if curr_idx != 0 and curr_idx % BATCH_SIZE == 0:
                    index_batch(batch)
                    print("DONE:IDX:" + str(curr_idx))
                prev_idx = curr_idx
        else:
            index_batch(batch)


if __name__ == '__main__':
    main()

