**Vulić and Moens (2015): [Bilingual Word Embeddings from Non-Parallel Document-Aligned Data Applied to Bilingual Lexicon Induction](http://www.aclweb.org/anthology/P15-2118)**
* Model induces bilingual word embeddings from non-parallel data **without** any other translation resources
* Focuses on learning lexicons from document-aligned comparable corpora (subject-aligned Wikipedia articles)
  * Spanish/English, Italian/English, Dutch/English
  * Retain only nouns that appear >4 times in the corpus
  * Use TreeTagger for POS tagging and lemmatization
* For a pair of aligned documents in a source and target language
  * The goal is to learn word embeddings for all words in both $$V^{S}$$ and $$V^{T}$$ such that the embeddings are semantically coherent and closely aligned over languages in a **single** embedding space
  * Merge two such aligned documents to form a single "pseudo-bilingual" document
    * Randomly shuffle the pseudo-bilingual document, to ensure that each word obtains surrounding context words from both languages
    * **The logic behind this leads to the conclusion that a larger context window size will lead to better embeddings**
    * Train a version of a monolingual skip-gram model on the shuffled pseudo-doc
    * The results show that computation is too expensive for extremely large windows, but flattens out with "reasonably large" windows
* Semantic similarity can be computed both monolingually and bilingually once the model is trained
  * Train using word2vec
    * learning rate=0.025, negative sampling=25, subsampling = $1e-4$, epochs=15
    * embedding dimensions={100, 200, 300, 40}
    * maximum window size={4, 8, 12,...,60}
  * Use cosine similarity for BLI task  
* Evaluate on 1000 "ground truth" one-to-one translation pairs for the three languages
  * Lexicon quality is best reflected in the $$Acc_{1}$$ score: the number of source language words from ground truth translation pairs for which the top-ranked word cross-lingually is the correct translation
  * Top words for monolingual similarity, cross-lingual similarity, and bilingual similarity are tested for a target word
