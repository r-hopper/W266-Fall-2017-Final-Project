# Final Project: Learning Crosslingual Word Embeddings
>   
_Roseanna Hopper, Mona Iwamoto, Maya Miller-Vedam_   
w266 : Natural Language Processing, Fall 2017    
UC Berkeley School of Information, MIDS Program  

**About this Project**
_For our final project we developed a Python implementation of a recently published paper discussing a strategy to learn crosslingual word embeddings without parallel corpora. This repo contains the code and suppporting files related to our work. Our final implmentation can be found [`models.py`](./Notebooks/models.py) and its use is demonstrated in the following Python Notebooks:_
* [English - Italian](./English-Italian_embeddings.ipynb)
* [English - Spanish](./English-Spanish_embeddings.ipynb)
* [English - Dutch](./English-Dutch_embeddings.ipynb)
* [English - Japonese](./English-Japonese_embeddings.ipynb)

## Quick Links
*  [Original Paper](https://arxiv.org/pdf/1606.09403.pdf) Duong, Long & Kanayama, Hiroshi & Ma, Tengfei & Bird, Steven & Cohn, Trevor. (2016). Learning Crosslingual Word Embeddings without Bilingual Corpora. 1285-1295. 10.18653/v1/D16-1136.
* [PanLex](https://panlex.org/) ; plus their [API Documentation](https://dev.panlex.org/api/)
* [Polyglot](http://polyglot.readthedocs.io/en/latest/modules.html) ; plus Rami Al-Rfou's [Demo & Tutorial](https://sites.google.com/site/rmyeid/projects/polyglot) including [Wikipedia Text Dumps](https://sites.google.com/site/rmyeid/projects/polyglot#TOC-Download-Wikipedia-Text-Dumps)
* [Word2Vec in TensorFlow Tutorial](http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/) from Adventures in Machine Learning.

## TO DO (reverse chronological order)
* Write up each model's architecture/development/interpretation/performance.
  > UPDATE 12/19 Done!
* Rewrite 'Lit Review' section to narrow the scope (just focus on answering the questions posed in the milestone)
  > UPDATE 12/15: Have written outlines for the answers to the milestone questions, and have tried to give more cohesion to the lit review section. Our reviews of the papers are fairly comprehensive, but need stronger/clearer ties back to Duong et al.
* Chose a set of Demo words to illustrate vector performance w/ cosine similarity.
* Evaluate all 3 models on the BLI
  > UPDATE 12/19 Done!
* Write the Bilingual Induction task Evaluation Code
  > UPDATE 12/15: evaluate_translation() (and the code to slot it into models.py) is written in a notebook, but needs to be incorporated and tested in models.py
* Create a third model that uses context to pick the word to substitute on
  > NOTE: ask Melody/James whether our interpretations make sense? In what sense is this "EM"?
* Create a second model with probabilistic word substitution
  > NOTE: this is just a matter of writting the code to count and store word frequencies then plug it into model1
* Create a first model with random word substitution (Maya)
  > UPDATE 12/12: I'm close... see [model1.py](./Notebooks/model1.py)
* Figure out the BLI task from Vulic & Moens (Roseanna)
  > UPDATE 12/10: Got data from Vulic for 3 languages, still need to write up the code to calculate accuracy/recall.
* Create DataParsing Functions (Maya)
  > UPDATE 12/12: Mostly done, see [parsing.py](./Notebooks/parsing.py) for `Corpus()` and `Vocabulary()` classes plus lazy batch generator for CBOW training.
  >> NEXT STEPS: Still having trouble initializing a `Vocabulary` from the full English Wiki data. Maybe an alternative strategy would be to strip the English sentences from Duong et Al's Italian-English raw data & just train with that?
* Work through [Word2Vec in TensorFlow Tutorial] & modify it to CBOW (Mona)
  > DONE 12/10: See (./Notebooks/Word2Vec_Tutorial_Notes.ipynb) and (./Notebooks/CBOW_implementation.ipynb)
* Load Raw Data into G Cloud Instance (Maya)
  > UPDATE 11/30: Done. See `/home/mmillervedam/Data` on the shared instance.
* Get C code running (Mona)
  > UPDATE 11/28: Compiled & running but direct conversion to Python does not seem like a good approach. Alternative options are to run the C code in python using a wrapper like in [this Word2Vec implementation](https://github.com/danielfrg/word2vec/blob/master/setup.py) or rebuild from scratch following Word2Vec code itself.
* Make a G-cloud instance we can all access (Maya)
  > DONE 11/30: You should be able to ssh in using `gcloud compute --project "synthetic-arc-178620" ssh --zone "us-west1-b" "finalproject" --ssh-flag "-L 8888:127.0.0.1:8888" --ssh-flag  "-L 6006:127.0.0.1:6006"`
* Literature Review about Crosslingual Word Embeddings(all)
  > ONGOING: see [Literature Review](./LiteratureReview) folder above.
* Draft Milestone due Nov 16 (Maya)
  > DONE 11/12: see [Deliverables](./Deliverables) folder above.
* Look for baseline model implementations in Python (Roseanna)
  > DONE 11/13: see [Baseline Models](./BaselineModels); they all appear to be in C.


## Other Links
* Open NMT [project website](http://opennmt.net/FAQ/#where-can-i-get-training-data-for-translation-from-x-to-x)
* OPUS [open source parallel corpus](http://opus.nlpl.eu) -- note that this cite also has a bunch of Machine Translation links.
* [Project Proposal G-Doc](https://docs.google.com/document/d/1KetDWpYzTtGK18eEqWcLiTbOCSH0ReBHH3oOpfCfLaI/edit)
* [Tensorflow NMT tutorial](https://research.googleblog.com/2017/07/building-your-own-neural-machine.html)

