from utils import load_idiom2tfidfs
from merge_idioms.service import build_mip


def main():
    idiom ="by hook or by crook"
    mip = build_mip()

    sentences = [
        "# When making your butch-buddy film, "
        "By Hook or By Crook, you and your cowriter,"
        " Silas Howard, decided that the butch characters would call each other"
        " ' he ' and ' him, ' but in the outer world of grocery stores and authority figures,"
        " people would call them ' she ' and ' her. '",  # 1
        "n. - Leaving home, escaping from their parents, from middle America and just come by hook or by crook to "
        "New York to be part of this scene. ",  # 2
        "She was a hustler and without a doubt, by hook or by crook, if she knew nothing else, "
        "she knew how to get money. ",  # 3
        " (gasps) Bingo! Flanders, are you willing to get Lovejoy back by hook or by crook? ", # 4
        "The movie makes you root for his white lie -- because politics for Lincoln was about passing laws"
        " by hook or by crook.",  # 5
        "The Maharashtra Congress on Tuesday attacked the BJP for grabbing 'power by hook or by crook' in Karnataka." # 6
    ]
    idiom2tfidfs = load_idiom2tfidfs()
    scores = list()
    for sent in sentences:
        tokens = [
            token.lemma_.lower()
            for token in mip(sent)
        ]
        score = idiom2tfidfs.eval_quality(idiom, tokens)
        scores.append(score)
    sents_with_scores = [
        (sent, score)
        for score, sent in zip(scores, sentences)
    ]
    # yeah.. if the sentence is too long..
    # you want a short, yet strong example sentence.
    # you also want to normalise the score.. right?
    # use sigmoid for that.
    sents_sorted = sorted(sents_with_scores, key=lambda x: x[1], reverse=True)
    for sent, score in sents_sorted:
        print(score, ":", sent)


if __name__ == '__main__':
    main()
