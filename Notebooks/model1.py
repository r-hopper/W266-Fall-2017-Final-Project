#!/usr/bin python
"""
Word2Vec with CBOW model that replaces centerword with
a randomly chosen translation from a provided dictionary.

INSPIRED BY: Duong, Long & Kanayama, Hiroshi & Ma, Tengfei & Bird,
    Steven & Cohn, Trevor. (2016). Learning Crosslingual Word
    Embeddings without Bilingual Corpora. 1285-1295. 10.18653/
    v1/D16-1136.

ADAPTED FROM: https://github.com/tensorflow/tensorflow/blob/r1.2/
    tensorflow/examples/tutorials/word2vec/word2vec_basic.py
"""

from __future__ import print_function
import os
import re
import collections
import numpy as np
from sklearn.manifold import TSNE


class SubstitutionModel(object):
    """
    Single Layer Neural Net to Learn Crosslingual Word Embeddings.

    Init Args:
                V   - (int) vocabulary size
                H   - (int) embedding size
        softmax_ns  - (int, default = 64) number of samples for softmax
            graph   - (optional) tf.Graph() instance
        learning_rate (float, default = 1.0)

    Methods:
        self.BuildCoreGraph()
        self.BuildTrainingGraph()
        self.BuildValidationGraph()
        self.
    """


    def __init__(self, graph=None, *args, **kwargs):
        """
        Initialize TensorFlow Neural Net Model.
        Args:
          V: vocabulary size
          H: embedding size

        Kwargs:
          softmax_ns = 64  (number of negative samples)
          alpha = 1.0  (learning rate)
          examples = np.array of 5 top 100 words for validation
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph # TODO : remove this unless we plan to init as tf.const
    def SetParams(self, V, H, softmax_ns=64, learning_rate=1.0):
        # Model structure.
        self.V = V
        self.H = H

        # Training hyperparameters
        self.softmax_ns = softmax_ns
        self.alpha = learning_rate

        # Words for validation
        self.examples = np.random.choice(100, 10, replace=False)

        # Results
        self.epochs_trained = 0
        self.final_embeddings = None

    @with_self_graph
    def BuildCoreGraph(self):

        batch_size = 128 # TODO : I've hard coded this for now b/c I want to
                         # get the rest of the code running, but eventually
                         # this should be inferred dynamically from the input
                         # shape as in a4.
        skip_window = 2  # TODO : ditto

        # Data Placeholders
        self.inputs_ = tf.placeholder(tf.int32, shape=[batch_size])
        self.context_ = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Embedding Layer
        with tf.variable_scope("Embedding_Layer"):
            self.embeddings_ = tf.Variable(tf.random_uniform([self.V, self.H],
                                            -1.0, 1.0), name='Embeddings')
            self.embed_ = tf.nn.embedding_lookup(self.embeddings_, self.inputs_)
            self.reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window*2)
            # Normalized Embeddings facillitate cosine similarity calculation
            # .... but don't train on these! they're just for evaluation!
            self.norm_ = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings_), 1, keep_dims=True))
            self.normalized_embeddings_ = self.embeddings_ / self.norm_

        # Hidden Layer
        with tf.variable_scope("Hidden_Layer"):
            self.W_ = tf.Variable(tf.truncated_normal([self.V, self.H],
                                  stddev=1.0 / math.sqrt(self.H)), name = 'W')
            self.b_ = tf.Variable(tf.zeros([self.V,], dtype = tf.float32), name = 'b')
            self.logits_ = tf.matmul(self.embed_, tf.transpose(self.W_)) + self.b_

    @with_self_graph
    def BuildTrainingGraph(self):
        with tf.variable_scope("Training"):
            nce_args = dict(weights=self.W_,
                            biases=self.b_,
                            labels=self.context_,
                            inputs=self.embed_,
                            num_sampled=self.softmax_ns,
                            num_classes=self.V)
            self.nce_loss_ = tf.reduce_mean(tf.nn.nce_loss(**nce_args))
            self.optimizer_ = tf.train.GradientDescentOptimizer(self.alpha)
            self.train_step_ = self.optimizer_.minimize(self.nce_loss_)


    @with_self_graph
    def BuildValidationGraph(self):
        self.test_ = tf.constant(self.examples, dtype=tf.int32)
        self.test_embed_ = tf.nn.embedding_lookup(self.normalized_embeddings_,
                                                  self.test_)
        self.similarity = tf.matmul(self.test_embed_,
                                    self.normalized_embeddings_,
                                    transpose_b=True)

    def learn_embeddings(self, num_steps, batch_fxn, data, index, verbose=True):
        """
        Runs a specified number of training steps.
        NOTE: right now the batch fxn is hard coded with inputs:
                  (data,batch_size=128,num_skips=2,skip_window=2)
              It should output two arrays representing the input &
              context indices for a single batch.
              TODO: replace this with something less clunky!
        """

        with tf.Session(graph=self.graph) as session:

            # initialize all variables
            init = tf.global_variables_initializer()
            init.run()
            print('... Model Initialized')
            if verbose:
                for var in tf.trainable_variables():
                    print("\t", var)

            # iterate through specificied number of training steps
            average_loss = 0
            for step in range(num_steps):
                # Get the next batch of inputs & their skipgram context
                batch_inputs, batch_context = batch_fxn(data, 128, 2, 2)

                # Run the train op
                feed_dict = {self.inputs_: batch_inputs, self.context_: batch_context}
                _, loss_val = session.run([self.train_step_, self.nce_loss_],
                                          feed_dict=feed_dict)

                # Logging Progress
                average_loss += loss_val
                loss_logging_interval = num_steps // 10
                sim_logging_interval = num_steps // 5
                if not verbose:
                    continue
                if step % loss_logging_interval == 0:
                    if step > 0:
                        average_loss /= loss_logging_interval
                    # The average loss is an estimate of the loss over the last 1000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
                if step % sim_logging_interval == 0:
                    sim = self.similarity.eval()
                    for i in xrange(len(self.examples)):
                        word = index[self.examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = '   Nearest to %s:' % word
                        for k in xrange(top_k):
                            nbr = index[nearest[k]]
                            log_str = '%s %s,' % (log_str, nbr)
                        print(log_str)
            # results
            self.epochs_trained = num_steps
            self.final_embeddings = self.normalized_embeddings_.eval()
        return self.final_embeddings

    def plot_embeddings_in_2D(self, num, index):
        """
        Plot 2D representation of embeddings.
        Args:
            num = int (number of examples to plot)
            index = reverse dictionary of word indices
            filename = path to save plot
        """
        if self.final_embeddings is None:
            print("You must train the embeddings before plotting.")
        else:
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs = tsne.fit_transform(self.final_embeddings[:num, :])
            labels = [index[i] for i in xrange(num)]
            plt.figure(figsize=(18, 18))  # in inches
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(str(label), xy=(x, y), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom')
            plt.show()
