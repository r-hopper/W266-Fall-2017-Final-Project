#!/usr/bin python
# -*- coding: utf-8 -*-
"""
Word2Vec with CBOW models that replaces centerword with
a translation from a provided dictionary.

Four classes:

    BiW2V:        - no word substitution
    BiW2V_random: - substitute a random translation
    BiW2V_mle:    - sustitute the translation that occurs
                    most often in the target language.
    BiW2V_nn:     - substitute the translation  word whose
                    context embedding is closest to to the
                    context vector from the source language.

INSPIRED BY:
    Duong, Long & Kanayama, Hiroshi & Ma, Tengfei & Bird,
    Steven & Cohn, Trevor. (2016). Learning Crosslingual Word
    Embeddings without Bilingual Corpora. 1285-1295. 10.18653/
    v1/D16-1136.

ADAPTED FROM:
    https://github.com/tensorflow/tensorflow/blob/r1.2/
    tensorflow/examples/tutorials/word2vec/word2vec_basic.py
AND: code provided as part of MIDS w266 assignment 4.
"""

from __future__ import print_function
import os
import math
import types
import collections
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE


# Helper function
def with_self_graph(function):
    """Decorator-foo borrowed from w266 a4."""
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

#####################################################################
##################### Model 0 - no word translation #################
class BiW2V(object):
    """
    Bilingual Word2Vec.

    Methods:
        self.BuildCoreGraph()
        self.BuildTrainingGraph(loss = 'sampled_softmax')
        self.BuildValidationGraph()
        self.translate(word_idxs)
        self.train(nSteps, data, sample = [3,4,5,6,7], verbose = True)
        self.plot_embeddings_in_2D(wordset)
    """

    def __init__(self, graph=None, *args, **kwargs):
        """
        Initialize TensorFlow Neural Net Model.
        Args:
          vocab - vocabulary object.
          H     - embedding size, an int.
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, vocab, H):
        # Vocabulary
        self.vocab = vocab

        # Model structure.
        self.V = self.vocab.size
        self.H = H

        # Hyperparameters
        with tf.variable_scope("Training_Parameters"):
            self.softmax_ns_ = 64 # TODO: find an alternative to hard coding?
            self.learning_rate_ = tf.placeholder_with_default(1.0, [],
                                                    name = 'learning_rate')

        # Results (parameters retrieved after training)
        self.context_embeddings = None
        self.word_embeddings = None


    @with_self_graph
    def BuildCoreGraph(self):
        """
        CBOW training model creates a reduced representation of
        the context window then passes it through an affine layer
        and into a softmax to predict the center word. In the
        bilingual version we additionally predict a translation of
        the centerword (Note: our embedding matrix represents the
        concatenated vocabularies of both source & target languages)
        """

        # Data Placeholders (note these will all be in batches)
        self.context_ = tf.placeholder(tf.int32, shape=[None, None])
        self.centerword_ = tf.placeholder(tf.int32, shape=[None,])
        self.translation_ = tf.placeholder(tf.int32, shape=[None,])

        # Embedding Layer
        with tf.variable_scope("Embedding_Layer"):
            # Random initialization NOTE: self.C_ is Duong's 'V'
            self.C_ = tf.Variable(tf.random_uniform([self.V, self.H],-1.0,1.0),
                                  name = 'ContextEmbeddings')

            # Input for hidden layer NOTE: self.input_ is Duong's 'h'
            embed = tf.nn.embedding_lookup(self.C_, self.context_)
            span = 8 # TODO: fix this so that it infers the context length!
            self.input_ = tf.div(tf.reduce_sum(embed, 1), span)


        # Hidden Layer
        with tf.variable_scope("Hidden_Layer"):
            # Random initialization NOTE: self.W_ is Duong's 'U'
            self.W_ = tf.Variable(tf.truncated_normal([self.V, self.H],
                                  stddev=1.0 / math.sqrt(self.H)),
                                  name = 'WordEmbeddings')
            self.b_ =tf.Variable(tf.zeros([self.V,],dtype=tf.float32), name='b')
            # hidden layer output = softmax input
            self.logits_ = tf.matmul(self.input_,tf.transpose(self.W_))+self.b_

        # No output layer because we don't intend to use this model, we just
        # want access to its parameters to use as features for other models.
        print("... TF graph created for BiW2V model.")


    @with_self_graph
    def BuildTrainingGraph(self):
        """
        Train Word & Context Embeddings (Duong's V and U) using
        Sampled Softmax to jointly optimize the probabilities of
        predicting the centerword & predicting its translation.
        """
        with tf.variable_scope("Training"):

            # print(self.W_.shape)
            # print(self.b_.shape)
            # print(self.input_.shape)
            # print(self.centerword_.shape)
            # self.labels_ = tf.expand_dims(self.centerword_, axis=1)
            # print('testing')
            # print(self.labels_.shape)

            # softmax for monolingual label
            mono_args = dict(weights = self.W_,
                             biases = self.b_,
                             inputs = self.input_,
                             labels = tf.expand_dims(self.centerword_, axis=1),
                             num_sampled = self.softmax_ns_,
                             num_classes = self.V)
            mono = tf.reduce_mean(tf.nn.sampled_softmax_loss(**mono_args))
            # softmax for crosslingual label
            cross_args = mono_args.copy()
            cross_args['labels'] = tf.expand_dims(self.translation_, axis=1)
            cross = tf.reduce_mean(tf.nn.sampled_softmax_loss(**cross_args))

            # loss function is their sum # TODO add regularizer
            # TODO: check if there is a way to pass 2 labels to tf.sftmx
            self.loss_ = mono #+ cross
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_)
            self.train_step_ = optimizer.minimize(self.loss_)

        print("... TF graph created for BiW2V training.")


    @with_self_graph
    def BuildValidationGraph(self):
        """
        Use cosine similarity to retrieve words with
        similar context or word embeddings.
        """
        with tf.variable_scope("Validation"):
            # words to validate
            self.valid_words_ = tf.placeholder(tf.int32, shape=[None,])

            # Normalized Embeddings facilitate cosine similarity calculation
            norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x),keep_dims=True))
            self.context_embeddings_ = self.C_ / norm(self.C_)
            self.word_embeddings_ = self.C_ / norm(self.W_)

            # TODO: add code here if we want to combine U and V w/ gamma weight

            # Retrieve context & word embeddings for validation words
            embedded_words = tf.nn.embedding_lookup(self.context_embeddings_,
                                                    self.valid_words_)
            self.similarity_ = tf.matmul(embedded_words,
                                         self.context_embeddings_,
                                         transpose_b=True)
        print("... TF graph created for BiW2V validation.")


    def translate(self, word_idxs):
        """
        Helper method used in training to translate centerwords at runtime.
        Base Implementation: no translation, this is a dummy method that
        will be overwritten in child classes.
        """
        return word_idxs


    def train(self, nSteps, data, sample = [3,4,5,6,7], learning_rate = None, verbose = True):
        """
        Train the model on the provided data.

        Args:
            nSteps  - number of training steps, an int.
            data    - a batch generator that windows over the corpus.
            sample  - (optional) iterable of word idxs whose neighbors to
                    display after each training step to track progress.
            verbose - (optional) boolean to control logging behavior.

        After training, results can be accessed via these class vars:
                self.final_context_embeddings
                self.final_word_embeddings
        """

        # perform some checks
        msg = 'ERROR: Please provide an integer for nSteps'
        assert type(nSteps) == type(1) and nSteps > 0, msg
        msg = 'ERROR: data should be a batch generator'
        assert type(data) ==  types.GeneratorType, msg
        msg = 'ERROR: sample should be a list of word indexes'
        assert sample is None or type(sample[0]) == type(1), msg

        # set up logging intervals (for verbose training)
        loss_logging_interval = max( 1, nSteps // 10)
        sim_logging_interval = max( 1, nSteps // 5)

        # proceed with training
        with tf.Session(graph=self.graph) as session:

            # initialize all variables
            init = tf.global_variables_initializer()
            init.run()
            print('... Model Initialized')
            if verbose:
                for var in tf.trainable_variables():
                    print("\t", var)

            # train in batches
            step = 0
            average_loss = 0
            print('... Starting Training')
            for batch, labels in data:

                # Run the train op
                feed_dict = {self.context_ : batch,
                             self.centerword_ : labels,
                             self.valid_words_ : sample,
                             self.translation_ : self.translate(labels)}
                if learning_rate is not None:
                    feed_dict[self.learning_rate_] = learning_rate

                _, loss_val = session.run([self.train_step_, self.loss_],
                                          feed_dict = feed_dict)

                # Logging Average Loss
                average_loss += loss_val
                if verbose and step % loss_logging_interval == 0:
                    average_loss /= loss_logging_interval
                    print('... STEP', step, ': Average Loss :', average_loss)
                    average_loss = 0

                # Logging Example Word Neighbors
                if verbose and step % sim_logging_interval == 0:
                    sim = session.run(self.similarity_, feed_dict = feed_dict)
                    for i in xrange(len(sample)):
                        word = self.vocab.index[sample[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = '   [%s] closest: ' % word
                        for k in xrange(top_k):
                            nbr = self.vocab.index[nearest[k]]
                            log_str = '%s %s,' % (log_str, nbr)
                        print(log_str)

                # check stopping criteria
                step += 1
                if step > nSteps:
                    print('... Training Complete')
                    break

            # results (extract parameters to class vars)
            self.context_embeddings = self.context_embeddings_.eval()
            self.word_embeddings = self.word_embeddings_.eval()


    def evaluate_prediction(self, i, sim, top_k, word):
        """
        Given example source words and the ground truth translations,
        evaluate the number of source language words for which one of the top three predictions is the correct translation
        (fuzzy measure adopted from stricter Vulic and Moens task which requires that one predicted translation is exactly correct)

        Takes:
        i: tracking the for-loop for calculation of nearest
        sim: most similar words from similarity_()
        top_k: the number of predictions we're checking for in the ground truth list
        word: the word we are interested in evaluating

        Returns:
        nearest: the top_k nearest neighbors of word
        valid_translation: tracks how many words in "nearest" are valid translations
            (range 0-k, restricted by the number of translations in the ground truth list)
        """
        valid_translation=0
        
        nearest = (-sim[((self.vocab.size - 3) / 2 + 3)+i, :]).argsort()[1:top_k + 1] #Take the nearest from the second half of the matrix (target language is second half)
        #nearest = (-sim[(len(sim)/2)+i, :]).argsort()[1:top_k + 1] #Take the nearest from the second half of the matrix (target language is second half)
        for k in range(top_k):
            close_word = self.vocab.index[nearest[k]]
            total_translations = (gtt[gtt[source_lang] == word])
            if close_word in total_translations[total_translations.columns[1]].values:
                valid_translation+=1
            else:
                valid_translation+=0

        return nearest, valid_translation


    def evaluate(self, ground_truth_translations, sample, verbose=True):
        """
        Args:
            ground_truth_translations: the dictionary of ground truth translations 
            sample: indexes of words to feed to similarity_()
            verbose: (optional) will print mean accuracy if true
        """
        #Define the feed dict
        feed_dict = {self.valid_words_ : sample}
        
        # Log validation word closest neighbors
        sim = session.run(self.similarity_, feed_dict = feed_dict)
        bli = self.evaluate_prediction()
        total_valid=[] #Track the total number of valid translations in the nearest k
        any_valid=[] #Track whether ANY of the nearest k were valid translations
        for i in xrange(len(sample)):
            word = self.vocab.index[sample[i]]
            top_k = 3  # number of nearest neighbors
            #source_lang = "en" #Hard-coding for testing; should be self.vocab.language[0]
            #target_lang = "it" #Hard-coding for testing; should be self.vocab.language[1]
            nearest, valid_translation = bli(i, sim, top_k, word)
            total_valid.append(valid_translation)

        #For any_valid, we need 0/1 to calculate the mean
        accuracy = {}
        for s in range(len(total_valid)):
            if total_valid[s] > 0:
                any_valid.append(1)
            else:
                any_valid.append(0)
            word = self.vocab.index[sample[s]]
            word_acc = (sum(any_valid) / (len(any_valid)))
            accuracy[(word)] = word_acc
            if verbose:
                print('Total successful translation rate: %d' % word_acc)
                print accuracy
            return accuracy


    def plot_embeddings_in_2D(self, wordset):
        """
        Plot 2D representation of embeddings.
        Args:
            wordset - list of words to plot
        """
        if self.context_embeddings is None:
            print("You must train the embeddings before plotting.")
        else:
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs = tsne.fit_transform(self.context_embeddings[:num, :])
            labels = [self.vocab.index[i] for i in xrange(num)]
            plt.figure(figsize=(18, 18))  # in inches
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(str(label), xy = (x, y), xytext = (5, 2),
                             textcoords = 'offset points', ha = 'right',
                             va = 'bottom')
            plt.show()


#####################################################################
##################### Model 1 - random translation ##################

class BiW2V_random(BiW2V):
    """
    Bilingual Word2Vec.
    This model trains embeddings in two languages by jointly
    optimizing the softmax probability of the source langauge
    centerword and a randomly chosen translation.
    """

    def __init__(self, bilingual_dict, *args, **kwargs):
        """
        Initialize TensorFlow Neural Net Model.
        Args:
          index     - vocabulary dict of {idx : word}.
          H         - embedding size, an int.
          bilingual_dict - multilingual dictionary where words
                      are mapped to 1 or more translations.
        """  
        # in addition to the normal Word2Vec args....
        super(BiW2V_random, self).__init__(*args, **kwargs)
        self.translations = bilingual_dict


    def translate(self, word_idxs):
        """
        Helper method to return the index of a randomly chosen
        translation for each word. If no translation is found,
        return the target language <unk> token.
        """
        target_words = []
        for idx in word_idxs:
            wrd = self.vocab.index[idx]
            translations = self.translations.get(wrd, ['<unk>'])
            target_words.append(np.random.choice(translations))
        return self.vocab.to_ids(target_words)
