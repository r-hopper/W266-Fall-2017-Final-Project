#!/usr/bin/env python
"""
Helper Functions for implementing Word2Vec in Python.

Most of the functions in this file come from the Official 
Tensorflow Docs and are made available via the word2vec
tutorial at: https://github.com/tensorflow/tensorflow/blob
/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
       
As noted, some of these helper fuctions were written or modified
by the authors of adventuresinmachinelearning.com as part of their
word-2-vec tutorial which closely follows the Tensorflow code.

I have also modified some to suit our use case.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import numpy as np
import tensorflow as tf


def read_data(filename):
    """
    Extract the file as a list of words.
    NOTE: this is modified from original function in TF  
    tutorialwhich expected a zipped input file.
    """
    with open(filename) as f:
        data = tf.compat.as_str(f.read()).split()
    return data


def build_dataset(words, n_words):
    """
    Process raw inputs into a dataset.
    Creates vocabulary from top n words indexed by rank.
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data_index = 0
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context