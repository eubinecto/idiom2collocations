import re

from spacy import load
from idiom2collocations.paths import NLP_MODEL

sent = ['And', 'because', 'it', 'was', '[IDIOM]', 'Major', 'League',
        'Baseball', 'and', 'the', 'player', "'s", 'association', 'says',
        'they', 'did', "n't", 'know', 'about', 'it', 'until', 'that', "'s",
        'baloney']


def main():
    nlp = load(NLP_MODEL)
    untokenised = " ".join(sent)
    nlp.tokenizer.add_special_case("[IDIOM]", [{"ORTH": "[IDIOM]"}])  # not supposed to tokenise this.
    tokens = nlp(untokenised)
    lemma2pos = [
        (token.text if token.text == "[IDIOM]" else token.lemma_, token.pos_)
        for token in nlp(untokenised)
        # do some cleanup here.
        # if not token.pos_ == "PROPN"
        if not token.like_num  # don't need numbers
        if not re.match(r'^[A-Z!@\-\(\)]$', token.text)  # don't need them.
        if not token.is_punct  # don't need punctuations
        # starting with... and ending with.
    ]
    print(lemma2pos)
    lemmas = [
        lemma
        for lemma, pos in lemma2pos
    ]
    idiom_idx = lemmas.index('[IDIOM]')


if __name__ == '__main__':
    main()
