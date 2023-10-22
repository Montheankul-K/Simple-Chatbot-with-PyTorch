from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
stemmer = PorterStemmer()


# Defind tokenize function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Defind stem function
def stem(word):
    return stemmer.stem(word.lower())


# Defind bag of words function
def bag_of_words(tokenized_sentence, all_words):
    # Stem
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    # Create zero array
    bag = np.zeros(len(all_words), dtype=np.float32)
    # Fill 1 in bag
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
