import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
import string

from util import dataloader

import pandas as pd
import numpy as np


ellipsis_matcher = re.compile(r'(\.{2,5})([^.]|$)|(\.{3})')
tweet_tokenizer = TweetTokenizer()


def lower(raw: str) -> str:
    """
    Returns a lowercase copy of the input text.

    :param raw: Text which will be turned to lowercase. String.
    :return: A lowercase copy of the input text. String.
    """
    return raw.lower()


def remove_punctuation(raw: str) -> str:
    """
    Returns a copy of the input text with the removed punctuation.

    :param raw: Text which from which the punctuation will be removed. String
    :return: A copy of the input text with the removed punctuation. String
    """
    return "".join([char for char in raw if char not in string.punctuation])


def replace_ellipses(raw: str) -> str:
    """
    Returns a copy of the input text with the replaced number of dots in ellipses.

    :param raw: Text from which the ellipses will be reduced. String
    :return: A copy of the input text with the reduced ellipses. String
    """
    return ellipsis_matcher.sub(r' ... \2', raw)


def count_character(characters):
    """
    Counts instances of given characters.

    :param characters: The characters to be counted in the raw text.
    :return: Character count in raw text
    """
    return lambda raw, tokenized: (raw, [character in characters for character in raw].count(True))


def count_ellipses(raw, tokenized):
    """
    Counts ellipses in raw text.

    :param raw: Raw text
    :param tokenized: Tokens received from tokenizer
    :return: Ellipsis count
    """
    return raw, len(ellipsis_matcher.findall(raw))


def split_ellipses(raw, processed):
    """
    Splits ellipses in processed text.

    :param raw: Raw text
    :param tokenized: Tokens received from tokenizer
    :return: Raw text and processed tokens with ellipses split
    """
    array = []
    for elem in processed:
        if elem.startswith("..."):
            array.extend(elem.split())
        else:
            array.append(elem)

    return raw, array


def preprocess_and_tokenize(dataset: pd.DataFrame, remove_punct: bool = True):
    """
    Preprocesses text data and returns a Podium dataset.

    :param dataset: Dataset to be preprocessed and tokenized, containing text and labels. Pandas DataFrame.
    :param text_name: The name of the text column in the dataset Pandas DataFrame, 'text' by default. String.
    :param label_name: The name of the label column in the dataset Pandas DataFrame, 'label by default. String.
    :param finalize: Determines if dataset is returned finalized or not, True by default. Boolean
    :param use_vocab: Determines if a vocabulary is used or not, True by default. Boolean
    :param vocab_size: Determines the max size of the vocabulary, if it is used, 10000 by default. Integer.
    :param remove_punct: Determines if punctuation is removed or not. Boolean.
    :param use_features: Determines if features are used or not. Boolean.
    :return: A Podium Dataset, preprocessed and tokenized, and a Podium Vocab if it is used.
    """

    dataset.loc[:, "input_text"].apply(lambda raw: lower(raw))
    dataset.loc[:, "input_text"].apply(lambda raw: replace_ellipses(raw))
    if remove_punct:
        dataset.loc[:, "input_text"].apply(lambda raw: remove_punctuation(raw))

    dataset['tokenized'] = dataset.apply(lambda row: tweet_tokenizer.tokenize(row['input_text']), axis=1)

    # TODO: potrebno je jos dodati dio u kojem se tokanizirani twwetovi numericaliziraju preko vocabulary
