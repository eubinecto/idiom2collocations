from tqdm import tqdm

from idiom2collocations.loaders import load_idiom2lemma2pos


def main():
    idiom2lemma2pos = load_idiom2lemma2pos()
    # this one is failing... but why?
    for idiom, lemma2pos in tqdm(idiom2lemma2pos):
        lemmas = [lemma.strip() for lemma, _ in lemma2pos]
        try:
            idiom_idx = lemmas.index('[IDIOM]')
        except ValueError:
            print('### {} ###'.format(idiom))
            print(lemma2pos)


if __name__ == '__main__':
    main()
