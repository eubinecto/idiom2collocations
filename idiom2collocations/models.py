import math
from functools import reduce
from typing import List, Tuple, Generator, Dict, Union
from functional.pipeline import Sequence
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from tqdm import tqdm
from nltk import FreqDist
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
        for idiom, x_idioms, idiom_xs, xx_idioms, idiom_xxs, xxx_idioms, idiom_xxxs in self.clusters():
            self.x_idiom[idiom] = FreqDist(x_idioms).most_common(len(x_idioms))
            self.idiom_x[idiom] = FreqDist(idiom_xs).most_common(len(idiom_xs))
            self.xx_idiom[idiom] = FreqDist(xx_idioms).most_common(len(xx_idioms))
            self.idiom_xx[idiom] = FreqDist(idiom_xxs).most_common(len(idiom_xxs))
            self.xxx_idiom[idiom] = FreqDist(xxx_idioms).most_common(len(xxx_idioms))
            self.idiom_xxx[idiom] = FreqDist(idiom_xxxs).most_common(len(idiom_xxxs))

    def clusters(self) -> Generator[List[Tuple[str, list, list, list, list]], None, None]:
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
        self.verb_colls: Dict[str, List[Tuple[str, Union[int, float]]]] = dict()
        self.noun_colls: Dict[str, List[Tuple[str, Union[int, float]]]] = dict()
        self.adj_colls: Dict[str, List[Tuple[str, Union[int, float]]]] = dict()
        self.adv_colls: Dict[str, List[Tuple[str, Union[int, float]]]] = dict()

    def fit(self):
        """
        fit the model. i.e. init idiom2colls dct.
        """
        raise NotImplementedError


class TFCollModel(CollocationModel):

    def fit(self):
        for idiom, _, verb_bow, noun_bow, adj_bow, adv_bow in self.idiom2bows:
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

    def __init__(self, idiom2bows: Sequence):
        super().__init__(idiom2bows)

        # get all the docs to build a dictionary
        docs = [
            list(verb_bow.keys()) + list(noun_bow.keys()) + list(adj_bow.keys()) + list(adv_bow.keys())
            for _, verb_bow, noun_bow, adj_bow, adv_bow in self.idiom2bows
        ]
        # build a dictionary. this will be used for Tfidf gensim model!
        self.dct = Dictionary(docs)

    def docs(self, pos: str) -> List[List[Tuple[int, int]]]:
        if pos == "VERB":
            idx = 1
        elif pos == "NOUN":
            idx = 2
        elif pos == "ADJ":
            idx = 3
        elif pos == "ADV":
            idx = 4
        else:
            raise ValueError
        # note the the docs are sorted by idioms.
        docs = self.idiom2bows.map(lambda x: (x[0], x[idx]))\
                              .group_by_key()\
                              .sorted(key=lambda x: x[0])\
                              .map(lambda x: x[1])
        # translate this into the bow's format.
        docs_reduced = (
            reduce(lambda a, b: a.update(b) or a, bows)
            for bows in docs
        )
        return [
            [(self.dct.token2id[lemma], count) for lemma, count in bow.items()]
            for bow in docs_reduced
        ]

    def fit(self):
        # group by, and fit the collocations for verbs
        idiom_keys = sorted(list(self.idioms))
        verb_docs = self.docs("VERB")
        noun_docs = self.docs("NOUN")
        adj_docs = self.docs("ADJ")
        adv_docs = self.docs("ADV")
        # fit the models.
        verb_tfidf = TfidfModel(verb_docs, smartirs='ntc')
        noun_tfidf = TfidfModel(noun_docs, smartirs='ntc')
        adj_tfidf = TfidfModel(adj_docs, smartirs='ntc')
        adv_tfidf = TfidfModel(adv_docs, smartirs='ntc')
        for idiom, res_verb, res_noun, res_adj, res_adv in \
                zip(idiom_keys, verb_tfidf[verb_docs], noun_tfidf[noun_docs], adj_tfidf[adj_docs], adv_tfidf[adv_docs]):
            self.update(self.verb_colls, idiom, res_verb)
            self.update(self.noun_colls, idiom, res_noun)
            self.update(self.adj_colls, idiom, res_adj)
            self.update(self.adv_colls, idiom, res_adv)

    def update(self, coll_dict: dict, idiom: str, res: List[Tuple[int, float]]):
        coll_dict[idiom] = sorted([(self.dct[idx], tfidf) for idx, tfidf in res],
                                    key=lambda x: x[1],
                                    reverse=True)


class PMICollModel(CollocationModel):
    """
    Point-wise Mutual Information.
    Needs some hyper parameter tuning.
    """

    def __init__(self, idiom2bows: Sequence, idiom2lemma2pos: Sequence, lower_bound: int = 0):
        super().__init__(idiom2bows)
        self.idiom2lemma2pos = idiom2lemma2pos  # need this to compute the occurrences.
        self.idiom2count: Dict[str, int] = dict()
        self.verb2count: Dict[str, int] = dict()
        self.noun2count: Dict[str, int] = dict()
        self.adj2count: Dict[str, int] = dict()
        self.adv2count: Dict[str, int] = dict()
        # the size of the entire vocab
        self.verb_n: int = 0
        self.noun_n: int = 0
        self.adj_n: int = 0
        self.adv_n: int = 0
        # the lower bound of the frequency
        self.lower_bound = lower_bound

    def fit(self):
        # compute the occurrences & vocab size.
        for idiom, lemma2pos in tqdm(self.idiom2lemma2pos):
            self.idiom2count[idiom] = self.idiom2count.get(idiom, 0) + 1
            for lemma, pos in lemma2pos:
                if lemma == "[IDIOM]":
                    continue
                if pos == "VERB":
                    self.verb2count[lemma] = self.verb2count.get(lemma, 0) + 1
                    self.verb_n += 1
                elif pos == "NOUN":
                    self.noun2count[lemma] = self.noun2count.get(lemma, 0) + 1
                    self.noun_n += 1
                elif pos == "ADJ":
                    self.adj2count[lemma] = self.adj2count.get(lemma, 0) + 1
                    self.adj_n += 1
                elif pos == "ADV":
                    self.adv2count[lemma] = self.adv2count.get(lemma, 0) + 1
                    self.adv_n += 1
        # compute the oc-occurrences
        idiom_verb_co = self.idiom_lemma_co('VERB')
        idiom_noun_co = self.idiom_lemma_co('NOUN')
        idiom_adj_co = self.idiom_lemma_co('ADJ')
        idiom_adv_co = self.idiom_lemma_co('ADV')
                
        # pos_co for the entire thing.  (lemma: count)
        # now compute the colls
        for idiom in self.idioms:
            verb_co = idiom_verb_co.get(idiom, dict())
            assert set(verb_co.keys()).issubset(set(self.verb2count.keys()))
            noun_co = idiom_noun_co.get(idiom, dict())
            assert set(noun_co.keys()).issubset(set(self.noun2count.keys()))
            adj_co = idiom_adj_co.get(idiom, dict())
            assert set(adj_co.keys()).issubset(set(self.adj2count.keys()))
            adv_co = idiom_adv_co.get(idiom, dict())
            assert set(adv_co.keys()).issubset(set(self.adv2count.keys()))
            self.verb_colls[idiom] = self.colls(idiom_verb_co.get(idiom, dict()), self.idiom2count[idiom],
                                                self.verb2count, self.verb_n)
            self.noun_colls[idiom] = self.colls(idiom_noun_co.get(idiom, dict()), self.idiom2count[idiom],
                                                self.noun2count, self.noun_n)
            self.adj_colls[idiom] = self.colls(idiom_adj_co.get(idiom, dict()), self.idiom2count[idiom],
                                               self.adj2count, self.adj_n)
            self.adv_colls[idiom] = self.colls(idiom_adv_co.get(idiom, dict()), self.idiom2count[idiom],
                                               self.adv2count, self.adv_n)

    def idiom_lemma_co(self, pos: str) -> Dict[str, Dict[str, int]]:
        """
        count idiom-lemma co-occurrences.
        """
        idiom_lemma_co = dict()
        if pos == "VERB":
            idx = 1
        elif pos == "NOUN":
            idx = 2
        elif pos == "ADJ":
            idx = 3
        elif pos == "ADV":
            idx = 4
        for idiom, pos_bows in tqdm(self.idiom2bows.map(lambda x: (x[0], x[idx])).group_by_key()):
            for pos_bow in pos_bows:
                for lemma, count in pos_bow.items():
                    bow_so_far = idiom_lemma_co.get(idiom, dict())
                    bow_so_far[lemma] = bow_so_far.get(lemma, 0) + count
                    idiom_lemma_co[idiom] = bow_so_far
        return idiom_lemma_co
    
    def colls(self, pos_co: Dict[str, int], idiom_count: int, pos2count: dict, n: int):
        return sorted([
            (lemma, self.pmi(p_x_y=count / n,  # this is the pos_co-occurrence!
                             p_x=idiom_count / n,
                             p_y=pos2count[lemma] / n))
            for lemma, count in pos_co.items()
            # should be more frequent than the lower bound.
            if count > self.lower_bound
            ],
            key=lambda x: x[1],
            reverse=True)
    
    @staticmethod
    def pmi(p_x_y: float, p_x: float, p_y: float) -> float:
        """
        point-wise mutual information
        I(x,y) = log(p(x,y)/(p(x)p(y)))
        = log(p(x, y)) - log(p(x) * p(y))
        = log(p(x, y)) - (log(p(x)) + log(P(y)))
        """
        log_p_x_y = math.log(p_x_y, 2)
        log_p_x = math.log(p_x, 2)
        log_p_y = math.log(p_y, 2)
        return log_p_x_y - (log_p_x + log_p_y)


