**Täckström et al. (2012): [Cross-lingual Word Clusters for Direct Transfer of Linguistic Structure](http://aclweb.org/anthology/N12-1052)**
>**Summary**:
* High level: Applying a model built on a resource-rich language to a resource-poor language allows us to get around the lack of a task specific annotated corpus in the latter. (premise: stuff like POS tagging or syntactic parse trees are needed for Machine Translation to perform well but we don't have this kind of information on a large scale for many languages)
* The downstream task here is "cross-lingual structure prediction" (_What does that mean practically?-- dependency parsing and named entity recognition?_)... they specify that they're measuring "the relative error of delexicalized dependency parsers, trained on English treebanks and transferred to foreign languages".
* At the time of writing, they report that, for most languages, efforts to do unsupervised parsing "is still well below that of supervised systems and recent work has established that the performance is also below simple methods of linguistic transfer (McDonald et al., 2011)"
* The first part of the paper is about a mono-lingual task: as best I understand their method instead of learning a continuous multi-dimensional for words in the corpus they are developing features which correspond to cluster identities developed from a semi-supervised task focusing on language structure... it sounds a little like the POS tagging model where they use emmission & transition probabilties except instead of a POS transition matrix the matrix captures transitions between these unsupervised "classes" (clusters).
* section 4 is where the cross lingual analysis starts ....
* Definition of transfer learning: "The aim of transfer methods is instead to use knowl- edge induced from labeled resources in one or more source languages to construct systems for target lan- guages in which no or few such resources are avail- able (Hwa et al., 2005). Currently, the performance of even the most simple direct transfer systems far exceeds that of unsupervised systems (Cohen et al., 2011; McDonald et al., 2011; Søgaard, 2011)."
* They cite their starting point as : "the delexicalized direct transfer method proposed by McDonald et al. (2011) based on work by Zeman and Resnik (2008)" -- sounds like they're using universal POS to model word relationships in one languge then using unsupervised learning in both languages to cluster similar words within & across langauges, finally they apply the structure learned in the first language to the second using the clusterings to form that transition matrix ( probabilities from the first language get applied to words in the second). Note that the second step -- training the clusterings, requires word aligned training data + some fancy footwork that I did not read thoroughly enough to understand. 
>**Notes/Questions**: What is a "delexicalized dependency parser"? (mentioned in the abstract) ... infact when we talk about 'lexical' features of words more generally... what does that mean?
>**Citation**: Oscar Ta ̈ckstro ̈m, Ryan McDonald and Jakob Uszkoreit. "Cross-lingual Word Clusters for Direct Transfer of Linguistic Structure" 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 477–487, Montre ́al, Canada, June 3-8, 2012. ⃝c 2012 Association for Computational Linguistics


**Klementiev (2012): [Inducing Crosslingual Distributed Representations of Words](http://www.aclweb.org/anthology/C12-1089)**
* Introducing a method to represent both source and target language (crosslingual word embeddings) as a dense vector in the same vector space
* Assigning similar embedding vectors to translations; method suffers from errors in automatic word alignment

**Mikolov et al. (2013): [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/pdf/1309.4168.pdf)**
* Introducing contextual bag of words model (CBOW): learning word representations based on distributional contexts
* Using a bilingual dictionary as a source of bilingual information (as opposed to identical bilingual texts)
* Filtering a bilingual dictionary for one-to-one translations sidesteps issues with noisy dictionaries, but discards information
* Using cascade-style training (embeddings in source and target language are learned separately and then combined), which appears less effective than joint training of embeddings
  * **Mikolov Notes**
  * Paper proposes a technique for automating the process of generating dictionaries and phrase tables for translation.
  * Method builds a monolingual model for a particular language based on a large body of text, and then creates a linear projection between languages using a "small" bilingual dictionary. 
  * Uses the CBOW model (combining the representations of surrounding words to predict the word in the middle), as CBOW can be trained on a large corpus in a short time due to low computational complexity.
  * Relies on linear transformation between languages, in which $x_{i} \in \R^{d_{i}}$  is the vector representation of word $i$, and we have a word pair $\{ x_{i}, z_{i} \} \stackanchor{n}{i=1}$. Here, we rely on finding a transformation matrix $W$ such that $Wx_{i}$ approximates $z_{i}$. 
  * Compares model performance to two baselines: (1) similarity of the morphological structure of words (e.g. edit distance), and (2) similarity of word co-occurrences.
  * They find that although this model can translate words with high frequency ranks with good precision, incorporating edit distance into the translation guess improves precision even further.

