#!/usr/bin python
"""
Data parsing classes & functions supporting our
w266 Project on Crosslingual Word Embeddings.

Acknowledgements:
    As noted, a number of these methods are derived
    from code provided by course instructors and/or
    the TensorFlow Tutorial on Word2Vec.
"""
from __future__ import print_function
import os
import re
import collections
import numpy as np


class Corpus(object):
    """
    Class with helper methods to read from a Corpus.
    Intended to facillitate working with multiple corpora at once.
    Init Args:
        path - (str) filepath of the raw data
        lang - (str) optional language prefix to prepend when reading
    Methods:
        gen_tokens - generator factory for tokens in order
        gen_sentences - generator factory for sentences in order
    """

    def __init__(self, path, lang = ''):
        err_msg = "ERROR: corpus filepath not valid"
        assert os.path.isfile(path), err_msg
        self.path = path
        self.lang = lang

    def gen_tokens(self):
        """Return a generator of tokens."""
        for line in open(self.path, 'rb'):
            for token in line.strip().lower().split():
                yield self.lang + '_' + token

    def gen_sentences(self):
        """Return a generator of sentences."""
        for line in open(self.path, 'rb'):
            line = line.lower().strip('\n')
            yield re.sub(' ', ' ' + self.lang + '_', ' ' + line)

class Vocabulary(object):
    """
    This class is based heavily on code provided in a4 of MIDS w266, Fall 2017.
    Init Args:
        tokens    - iterable of tokens to count
        wordset   - (optional) limit vocabulary to these words
        size      - (optional) integer, number of vocabulary words
    Attributes:
        self.index   - dictionary of {id : type}
        self.size    - integer, number of words in total
        self.types   - dictionary of {type : id}
        self.wordset - set of types
    Methods:
        self.to_ids(words) - returns list of ids for the word list
        self.to_words(ids) - returns list of words for the id list
        self.sentence_to_ids(sentence) - returns list of ids with start & end
    """

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    START_ID = 0
    END_ID = 1
    UNK_ID = 2

    def __init__(self, tokens, wordset = None, size=None):
        # Count tokens from corpus (filter if wordset was specified)
        keep = lambda x: x in wordset if wordset is not None else True
        counts = collections.Counter([t for t in tokens if keep(t)])
        top_counts = counts.most_common(None if size is None else (size - 3))
        types = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] +
                 [w for w,c in top_counts])

        # Easy access to various formats:
        self.wordset = set(types)
        self.index = dict(enumerate(types))
        self.types = {v:k for k,v in self.index.iteritems()}
        self.size = len(self.index)
        if size is not None:
            assert(self.size <= size)

    def to_ids(self, words):
        return [self.types.get(w, self.UNK_ID) for w in words]

    def to_words(self, ids):
        return [self.index.get(i, self.UNK_TOKEN) for i in ids]

    def sentence_to_ids(self, sentence):
        return [self.START_ID] + self.to_ids(sentence.split()) + [self.END_ID]


def batch_generator(corpus, vocabulary, batch_size, bag_window, max_epochs = None):
    """
    Function to iterate repeated over a corpus delivering
    batch_size arrays of ids and context_labels for CBOW.

    Args:
        corpus - an instance of Corpus()
        vocabulary - an instance of Vocabulary()
        batch_size - int, number of words to serve at once
        bag_window - context distance for CBOW training
        max_epochs - int(default = None) stop generating

    Yields:
        batch: np.array of dim: (batch_size, 2*bag_window)
               Represents set of context words.
        labels: np.array of dim: (batch_size, 1)
               Represents center words to predict/translate.

    WARNING: this generator will go on ad infinitum unless
    you specify max_epochs or explicitly break.
    """
    nEpochs = 0
    span = 2 * bag_window + 1 # context size
    batch = [] # context lists of len span - 1
    labels = [] # center words

    # loop through corpus sentences
    data_generator = corpus.gen_sentences()
    ids = collections.deque()
    contexts = collections.deque()
    while True:
        try: # load the next sentence
            sentence = next(data_generator)
        except StopIteration: # or reload corpus
            if nEpochs < max_epochs or max_epochs is None:
                nEpochs += 1
                data_generator = corpus.gen_sentences()
                sentence = next(data_generator)
            else:
                print("... ERROR: Max Epochs (%s) Reached." % max_epochs)
                break

        # get the ids & their contexts
        tokens = vocabulary.sentence_to_ids(sentence)
        ids.extend(tokens[1:-1])
        contexts.extend([tokens[i - bag_window : i] +
                         tokens[i + 1 : i + bag_window + 1]
                         for i in range(1, len(tokens)-1)])

        # emit a batch if you can
        err_msg = "... ERROR: ids/context length mismatch"
        assert len(ids) == len(contexts), err_msg
        while len(ids) >=  batch_size:
            batch = [contexts.popleft() for _ in range(batch_size)]
            labels = [ids.popleft() for _ in range(batch_size)]
            yield batch, labels
