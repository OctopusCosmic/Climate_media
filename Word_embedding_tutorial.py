# video source: https://www.youtube.com/watch?v=Ouwe_Mnz0z0
# notebook source: https://github.com/PythonWorkshop/intro-to-nlp-with-pytorch/blob/master/Word%20Embeddings/Word%20Embeddings.ipynb


import torch
from torch.nn.functional import one_hot
def trigram_example(): # use former two words to predict the third words
    # We will use Shakespeare Sonnet 2
    test_sentence = """Tomorrow, and tomorrow, and tomorrow,
        Creeps in this petty pace from day to day,
        To the last syllable of recorded time;
        And all our yesterdays have lighted fools
        The way to dusty death. Out, out, brief candle!
        Life's but a walking shadow, a poor player,
        That struts and frets his hour upon the stage,
        And then is heard no more. It is a tale
        Told by an idiot, full of sound and fury,
        Signifying nothing.
        """.lower().split()
    CONTEXT_SIZE = 2 # use former two words to train and predict the third one
    # Build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]
    # print the first 3, just so you can see what they look like
    print(len(trigrams))



def main():

    # text message or sentences
    sentence = "Cain also said the customers involved would pay premium rates for the special generation and that the cost of the projects would not be passed on to the rest of Alabama Power's customer base. "
    # get all words
    words = sentence.lower().split(" ")
    print(f'words: {words}')
    # get all unique words as vocabulary
    vocab1 = list(set(words))
    print(f'vocab1: {vocab1}')
    # convert vocabulary to indexes
    word_to_ix1 = {}
    for i, word in enumerate(vocab1):
        word_to_ix1[word] = i
    print(word_to_ix1)
    # encode the vocabulary as tensors using indexes
    words = torch.tensor([word_to_ix1[w] for w in vocab1], dtype=torch.long)
    one_hot_encoding = one_hot(words)
    print(vocab1)
    print(one_hot_encoding)

main()



