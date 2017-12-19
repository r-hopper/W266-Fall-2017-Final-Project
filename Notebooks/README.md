## Code Files:
* __`parsing.py`__: This file contains classes and functions for parsing and preprocessing bilingual corpora, dictionaries and vocabularies. It also contains the `batch-generator` which is the expected input format for our model classes. 

* __`models.py`__: This file contains our main class `BiW2V` which implements a bilingual version of the Word2Vec Algorithm. Its three subclasses, `BiW2V_random`, `BiW2V_mle` and `BiW2V_nn` represent three different ways to handle the bilingual signal during training.  

Please see docstrings for more detailed information about each file including usage and citations.

## Notebooks in Chronological Order:
1. [Data Exploration]("./Data Exploration.ipynb"): _First look at provided data plus info on how to acquire Wiki Dump files from Rami Al Rfou's website._
2. [Word2Vec Tutorial Notes](./Word2Vec_Tutorial_Notes.ipynb) _Notes & code from TensorFlow tutorial and Adventures In Machine Learning Blog Post._
3. [Dictionary EDA](./Dictionary_EDA.ipynb)_Simple stats on the Panlex dictionaries from Duong et al._
4. [Data Parsing](./Data_Parsing) _Work on modifying w2v data processing functions to fit our data... this includes tokenizing & writing a batch iterator following the example form a4. Includes preprocessing of full data sets._
5. [CBOW_Implementation](./CBOW_implementation.ipynb) _Continuation from Word2Vec. Implemented generate_batch_cbow. Tested against skip-gram._ _Modified TensorFlow graph to use CBOW. Augmented graph with comparison of Full Softmax, NCE and Sampled Softmax. Spoiler alert: Sample Softmax wins. TODO: Update the class to do the same.  Next step, begin dictionary replacement._
6. [Ground Truth Translations](./Ground_Truth_Translations.ipynb) _Processes raw ground truth translation txt files from MUSE. Also contains the evaluate_prediction() function. Next step, integrate evaluate_prediction() with models.py. IMPORTANT: the cell where we define evaluate_prediction(source_lang, target_lang, top_k, word) and the immediately preceding cell ("# This is an example of...") are the best version of the function and describe its role within models.py._
7. [Simple Substitution Embeddings](./Simple_Substitution_Embeddings.ipynb) _Testing basic `BiW2V` class from `models.py` which trains bilingual word embeddings using a randomly choses translation to substitute for the centeword._
8. [Dictionaries & Vocabularies.ipynb]("./Dictionaries & Vocabularies.ipynb") _Training Bilingual Vocabularies and loading panlex dictionaries on fullsize shuffled corpora for each language pair so that they can be easily loaded & used in training._
9. [MLE Substitution_Embeddings](./MLE_Substitution_Embeddings.ipynb) _Testing `BiW2V_mle` class from `models.py` which trains bilingual word embeddings using the highest ranked translation to substitute for the centeword. Trains both random & MLE embeddings with 600K iterations and a learning rate of 0.5 on en-es corpus.This file also includes a demo of how to reload saved embeddings & use them to compute word similarity._
10. [CSim Substitution_Embeddings](./CSim_Substitution_Embeddings.ipynb) _Testing `BiW2V_nn` class from `models.py` which trains bilingual word embeddings using the highest ranked translation to substitute for the centeword. Trains both random & MLE embeddings with 600K iterations and a learning rate of 0.5 on en-es corpus._
