from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel


@dataclass
class Idiom2Tfidfs:
    idiom2tfidfs: Dict[str, Dict[str, float]]

    def __getitem__(self, idiom: str) -> Dict[str, float]:
        return self.idiom2tfidfs[idiom]

    def eval_quality(self, idiom: str, tokens: List[str]) -> float:
        """
        this is the one that evaluates the quality of a sentence (tokenised)
        :param idiom:
        :param tokens:
        :return:
        """
        scores: List[float] = [
            # if the token does not exist in the data, the score is zero
            self.idiom2tfidfs[idiom].get(token, 0)
            for token in tokens
        ]
        total = sum(scores)
        # TODO: I somehow have to encode: the shorter, the better. - isn't this...
        # what BM25 does?
        return np.tanh(total)  # use hyperbolic tangent


@dataclass
class Idiom2Lda:
    dct: Dictionary
    lda_model: LdaModel
    idiom2lda: Dict[str, List[float]]

    def eval_quality(self, idiom: str, tokens: List[str]) -> float:
        bow = self.dct.doc2bow(tokens)
        bow_lda = self.lda_model.get_document_topics(bow)
        # wait... so this is not a probability dist. it does not sum to 1.
        idiom_lda = self.idiom2lda[idiom]
        # measure the cosine similarity?