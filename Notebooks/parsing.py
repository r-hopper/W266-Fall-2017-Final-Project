#!/usr/bin python
# -*- coding: utf-8 -*-
"""
Data parsing classes & functions supporting our
w266 Project on Crosslingual Word Embeddings.
Note: Start Jupyter with
      jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000

Acknowledgements:
    As noted, a number of these methods are derived
    from code provided by course instructors and/or
    the TensorFlow Tutorial on Word2Vec.
"""
from __future__ import print_function
import os
import re
import time
import resource
import collections
import numpy as np


class Corpus(object):
    """
    Class with helper methods to read from a Corpus.
    Intended to facilitate working with multiple corpora at once.
    Init Args:
        path - (str) filepath of the raw data
        prefix - (str) optional language prefix to prepend when reading
    Methods:
        gen_tokens - generator factory for tokens in order
        gen_sentences - generator factory for sentences in order
    """

    def __init__(self, path, language = None):
        err_msg = "ERROR: corpus filepath not valid"
        assert os.path.isfile(path), err_msg
        self.path = path
        self.lang = language
        self.pre = '%s_'%(self.lang) if self.lang is not None else ''
        self.splits = 0    # for shuffle

    def gen_tokens(self):
        """Return a generator of tokens."""
        for line in open(self.path, 'rb'):
            for tok in line.strip().lower().split():
                yield self.pre + tok

    def gen_sentences(self):
        """Return a generator of sentences."""
        for line in open(self.path, 'rb'):
            line = line.lower().strip('\n')
            yield re.sub(' ', ' ' + self.pre, ' ' + line)

    def split_file(self, min_length = 1):
        '''
        Splits the file into smaller files of 10K sentences
        to be shuffled.  Warning this can be time/memory consuming.
        '''
        start = time.clock()
        PTH = './split_files/'
        newfile = True
        for line in open(self.path, 'rb'):
            if newfile == True:
                file = open(PTH+self.pre+"text_"+ str(self.splits) +".txt","w+")
                idx = 1
                newfile = False
            line = line.lower().strip('\n')
            wordcnt = len(line.strip().split(' '))
            # Only use sentence of length > min_length
            if wordcnt >= min_length:
                s = re.sub(' ', ' ' + self.pre, ' ' + line)
                file.write(s+'\n')
                idx +=1
            if idx > 10000:
                self.splits +=1
                file.close()
                newfile = True
        file.close()
        end = time.clock()
        print("Time to split - ", end-start, "seconds")
        print(self.splits, "files written")


    def draw_random(self, sample_size = 5000000):
        '''
        Randomly draws from the split files to create a shuffled
        monolingual file. Default sample size is 5M sentences as
        in the original paper.
        '''
        PTH = './split_files/'
        PTH_SHFL = './shuffled_files/'

        # We will be sampling 5M sentences.
        num_sampled = sample_size
        start = time.clock()
        errors = 0
        # We have a large number of open files.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert soft>9000
        print(resource.getrlimit(resource.RLIMIT_NOFILE))

        idx = 0
        f = [open(PTH+self.pre+"text_%d.txt" % i, "rU") for i in range(self.splits+1)]

        # This array stores the randomly chosen index of the file
        picklist = np.random.randint(0, self.splits+1, size = num_sampled)

        # Write to file
        out = open(PTH_SHFL+self.pre+"shuffled.txt", 'w+')
        for i in range(num_sampled):
            f_index = picklist[i]
            try:
                s = f[f_index].readline()

            except EOFError:
                errors += 1
                continue
            out.write(s)

        out.close()
        end = time.clock()
        print("Time to shuffle - ", end-start, "seconds")
        print(errors, "sentences skipped")

        for fh in f:
            fh.close()

def make_bilingual(corpus1,corpus2):
    '''
    Combines two shuffled corpora
    '''
    PTH_SHFL = './shuffled_files/'

    with open(PTH_SHFL+corpus1.pre+"shuffled.txt","r") as c1, open(PTH_SHFL+corpus2.pre+"shuffled.txt","r") as c2, open(PTH_SHFL+corpus1.pre+corpus2.pre+"shuf.txt","w") as out:
        i = 0
        for line in c1:
            out.write(line)
            try:
                line2 = c2.readline()
            except:
                continue
            out.write(line2)


class Vocabulary(object):
    """
    This class is based heavily on code provided in a4 of MIDS w266, Fall 2017.

    Attributes:
        self.index   - dictionary of {id : type}
        self.size    - integer, number of words in total
        self.types   - dictionary of {type : id}
        self.wordset - set of types
        self.language- order of languages in the index
                       (Bilingual version only)
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
        """
        Select Vocab from most Frequent Words
        Args:
            tokens    - iterable of tokens to count
            wordset   - (optional) limit vocabulary to these words
            size      - (optional) integer, number of vocabulary words
        """
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
    
    def load_from_index(self, index_dict):
        self.wordset = set(index_dict.keys())
        self.index = index_dict
        self.types = {v:k for k,v in self.index.iteritems()}
        self.size = len(self.index)

    def to_ids(self, words):
        return [self.types.get(w, self.UNK_ID) for w in words]

    def to_words(self, ids):
        return [self.index.get(i, self.UNK_TOKEN) for i in ids]

    def sentence_to_ids(self, sentence):
        return [self.START_ID] + self.to_ids(sentence.split()) + [self.END_ID]

class BilingualVocabulary(Vocabulary):
    """
    Vocabulary class with words from two languages.
    Inherits from Vocabulary and has all the same methods.
    """
    def __init__(self, tokens, languages, wordset = None, size=None):
        """
        Select Vocab from most Frequent Words in each Language
        Args:
            tokens    - iterable of tokens to count
            languages - tuple of language prefixes
            wordset   - (optional) limit vocab to these words
            size      - (optional) integer, number of vocab
                        words to include from each language.
        """
        # Class attribute
        self.language = languages
       
        # helper function for filtering wordset words
        keep = lambda x: x in wordset if wordset is not None else True

        # counter for each language
        counts_lang1 = collections.Counter()
        counts_lang2 = collections.Counter()

        # stream corpus and add counts
        for t in tokens:
            if keep(t):
                if t.startswith(self.language[0]):
                    counts_lang1[t] += 1
                elif t.startswith(self.language[1]):
                    counts_lang2[t] += 1
                else:
                    print("We are only bilingual at the moment.")

        # collect top words from each languages
        top_counts = counts_lang1.most_common(None if size is None else size)
        top_counts += counts_lang2.most_common(None if size is None else size)
        types = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] +
                 [w for w,c in top_counts])

        # Easy access to various formats:
        self.wordset = set(types)
        self.index = dict(enumerate(types))
        self.types = {v:k for k,v in self.index.iteritems()}
        self.size = len(self.index)
        if size is not None:
            assert(self.size <= size * 2 + 3)


def batch_generator(corpus, vocabulary, batch_size, window, max_epochs = None):
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
    span = 2 * window + 1 # context size
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
        tokens = [vocabulary.START_ID] * window  # left pad
        tokens += vocabulary.sentence_to_ids(sentence)[1:-1]
        tokens += [vocabulary.END_ID] * window  # right pad
        ids.extend(tokens[window: - window])
        contexts.extend([tokens[i - window : i] +
                         tokens[i + 1 : i + window + 1]
                         for i in range(window, len(tokens) - window)])

        # emit a batch if you can
        err_msg = "... ERROR: ids/context length mismatch"
        assert len(ids) == len(contexts), err_msg
        while len(ids) >=  batch_size:
            batch = [contexts.popleft() for _ in range(batch_size)]
            labels = [ids.popleft() for _ in range(batch_size)]
            yield batch, labels
