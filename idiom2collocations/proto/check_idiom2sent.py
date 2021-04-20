from tqdm import tqdm
from idiom2collocations.loaders import load_idiom2sent


def main():
    # idiom2sent is fine...
    idiom2sents = load_idiom2sent()
    for idiom, sent in tqdm(idiom2sents):
        idiom_idx = sent.index('[IDIOM]')
        if idiom == "look_the_other_way":
            should_include = {
                'and', 'because', 'player', 'association'
            }
            if should_include.issubset(set(sent)):
                print(sent)


if __name__ == '__main__':
    main()
