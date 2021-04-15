from typing import List, Tuple, Generator, Dict, Union
from functional.pipeline import Sequence
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from tqdm import tqdm
from nltk import FreqDist
from nltk.collocations import (
    BigramCollocationFinder,
    BigramAssocMeasures,
    TrigramCollocationFinder,
    TrigramAssocMeasures
)
from nltk.stem import WordNetLemmatizer


class ClusterModel:

    def __init__(self, idiom2sent: Sequence):
        self.idiom2sent = idiom2sent
        self.idioms = set([idiom for idiom, _ in idiom2sent])
        self.x_idiom: Dict[str, List[Tuple[tuple, int]]] = dict()
        self.idiom_x: Dict[str, List[Tuple[tuple, int]]] = dict()
        self.xx_idiom: Dict[str, List[Tuple[tuple, int]]] = dict()
        self.idiom_xx: Dict[str, List[Tuple[tuple, int]]] = dict()
        self.xxx_idiom = dict()
        self.idiom_xxx = dict()
        self.lemmatiser = WordNetLemmatizer()

    def fit(self):
        """
        just simple frequency counts of the tuples.
        """
        for idiom, x_idioms, idiom_xs, xx_idioms, idiom_xxs, xxx_idioms, idiom_xxxs in self.idiom2ngrams():
            self.x_idiom[idiom] = FreqDist(x_idioms).most_common(len(x_idioms))
            self.idiom_x[idiom] = FreqDist(idiom_xs).most_common(len(idiom_xs))
            self.xx_idiom[idiom] = FreqDist(xx_idioms).most_common(len(xx_idioms))
            self.idiom_xx[idiom] = FreqDist(idiom_xxs).most_common(len(idiom_xxs))
            self.xxx_idiom[idiom] = FreqDist(xxx_idioms).most_common(len(xxx_idioms))
            self.idiom_xxx[idiom] = FreqDist(idiom_xxxs).most_common(len(idiom_xxxs))

    def idiom2ngrams(self) -> Generator[List[Tuple[str, list, list, list, list]], None, None]:
        for idiom, sents in self.idiom2sent.group_by_key():
            sents: List[List[str]]
            # these are the ngrams to collect.
            x_idioms: List[Tuple[str, str]] = list()  # bigram 1
            idiom_xs: List[Tuple[str, str]] = list()  # bigram 2
            xx_idioms: List[Tuple[str, str, str]] = list()  # trigram 2
            idiom_xxs: List[Tuple[str, str, str]] = list()  # trigram 2
            xxx_idioms = list()
            idiom_xxxs = list()
            for sent in sents:
                # find the index of [IDIOM] token.
                idiom_idx = sent.index("[IDIOM]")
                # lemmatise the sentence
                sent = [self.lemmatiser.lemmatize(token) for token in sent]
                for idx, token in enumerate(sent):
                    if idx == idiom_idx - 1:
                        x_idioms.append((token, "[IDIOM]"))
                        continue
                    elif idx == idiom_idx + 1:
                        idiom_xs.append(("[IDIOM]", token))
                        continue
                    elif idx == idiom_idx - 2:
                        xx_idioms.append((sent[idx], sent[idx + 1], "[IDIOM]"))
                        continue
                    elif idx == idiom_idx + 2:
                        idiom_xxs.append(("[IDIOM]", sent[idx - 1], sent[idx]))
                        continue
                    elif idx == idiom_idx - 3:
                        xxx_idioms.append((sent[idx], sent[idx + 1], sent[idx + 2], "[IDIOM]"))
                    elif idx == idiom_idx + 3:
                        idiom_xxxs.append(("[IDIOM]", sent[idx - 2], sent[idx - 1], sent[idx]))
            yield idiom, x_idioms, idiom_xs, xx_idioms, idiom_xxs, xxx_idioms, idiom_xxxs


class CollocationModel:

    def __init__(self, idiom2bows: Sequence):
        """
        get it as a sequence object.
        """
        self.idiom2bows = idiom2bows
        self.idioms = set([idiom for idiom, _, _, _, _ in idiom2bows])  # must be a set.
        # --- these are the target collocations --- #
        self.verb_colls: Dict[str, List[Tuple[tuple, Union[int, float]]]] = dict()
        self.noun_colls: Dict[str, List[Tuple[tuple, Union[int, float]]]] = dict()
        self.adj_colls: Dict[str, List[Tuple[tuple, Union[int, float]]]] = dict()
        self.adv_colls: Dict[str, List[Tuple[tuple, Union[int, float]]]] = dict()

    def fit(self):
        """
        fit the model. i.e. init idiom2colls dct.
        """
        raise NotImplementedError


class TFCollModel(CollocationModel):

    def fit(self):
        for idiom, verb_bow, noun_bow, adj_bow, adv_bow in self.idiom2bows:
            self.verb_colls[idiom] = sorted(
                [(lemma, count) for lemma, count in verb_bow.items()],
                key=lambda x: x[1],
                reverse=True
            )
            self.noun_colls[idiom] = sorted(
                [(lemma, count) for lemma, count in noun_bow.items()],
                key=lambda x: x[1],
                reverse=True
            )
            self.adv_colls[idiom] = sorted(
                [(lemma, count) for lemma, count in adj_bow.items()],
                key=lambda x: x[1],
                reverse=True
            )
            self.adj_colls[idiom] = sorted(
                [(lemma, count) for lemma, count in adv_bow.items()],
                key=lambda x: x[1],
                reverse=True
            )


class TFIDFCollModel(CollocationModel):

    def fit(self):
        pass


class PMICollModel(CollocationModel):
    """
    Point-wise Mutual Information.
    Needs some hyper parameter tuning.
    """
    def fit(self):
        # TODO: fix this.
        bigram_coll_finder: BigramCollocationFinder = BigramCollocationFinder.from_documents(self.get_docs())
        bigram_measures = BigramAssocMeasures()
        bigram_coll_finder.apply_freq_filter(4)
        # look, this  will take ages, my god.
        for bigram, score in tqdm(bigram_coll_finder.score_ngrams(bigram_measures.pmi)):
            if bigram[0] in self.idioms:
                idiom = bigram[0]
                idiom_x = (idiom, bigram[1], score)
                print(idiom_x)
                self.idiom_x_colls[idiom] = self.idiom_x_colls.get(idiom, list()).append(idiom_x)
            if bigram[1] in self.idioms:
                idiom = bigram[1]
                x_idiom = (bigram[0], idiom, score)
                print(x_idiom)
                self.x_idiom_colls[idiom] = self.x_idiom_colls.get(idiom, list()).append(x_idiom)

    def get_docs(self) -> Generator[List[str], None, None]:
        for idiom, sent in tqdm(self.idiom2contexts):
            sent[sent.index("[IDIOM]")] = idiom
            yield sent
