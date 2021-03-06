
# Contents at a Glance
* [**Tackstrom et al. (2012)**](http://aclweb.org/anthology/N12-1052), *Cross-lingual Word Clusters for Direct Transfer of Linguistic Structure* ([notes](./Tackstrom-et-al-2012.md)): transfer learning of language structure using features developed from bilingual word clusters -- precursor to crosslingual word embeddings.
* [**Klementiev et al. (2012)**](http://www.aclweb.org/anthology/C12-1089), *Inducing Crosslingual Distributed Representations of Words* ([notes](./Klementiev-et-al-2012.md)): Introduces a method to represent both source and target language (crosslingual word embeddings) as a dense vector in the same vector space. Assigning similar embedding vectors to translations; method suffers from errors in automatic word alignment.
* [**Levy and Goldberg (2014)**](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization), *Neural Word Embedding as Implicit Matrix Factorization* ([notes](./Levy-Goldberg-2014.md)): Explores the mathematical basis of skip-gram with negative sampling (as introduced by Mikolov et al.). They conclude that the matrices V and U (as Duong et al. would notate them) represent the PMI between a word-context pair.
* [**Mikolov et al. (2013)**](https://arxiv.org/pdf/1309.4168.pdf), *Exploiting Similarities among Languages for Machine Translation* ([notes](./Mikolov-2012.md)): Introducing contextual bag of words model (CBOW): learning word representations based on distributional contexts with a bilingual dictionary as a source of bilingual information (as opposed to identical bilingual texts).
* [**Rfou et al. (2013)**](http://www.aclweb.org/anthology/W13-3520), *Polyglot: Distributed Word Representations for Multilingual NLP* ([notes](./Rfou-et-al-2013.md)): Introduces Polyglot crosslingual word embeddings & assesses their performance on the POS task. The authors' data are used by Duong et al. and their embeddings are publicly available.
* [**Vulić and Moens (2013)**](http://www.aclweb.org/anthology/P15-2118), *Cross-Lingual Semantic Similarity of Words as the Similarity of Their Semantic Word Responses* ([notes](./Vulic-Moens-2013.md), [data](https://github.com/r-hopper/W266-Fall-2017-Final-Project/tree/master/BaselineModels/data)): Describes how they arrive at the 1000 "ground truth" translation pairs for the BLI task evaluation and also how they calculate "accuracy". 
* [**Vulić and Moens (2015)**](http://www.aclweb.org/anthology/P15-2118), *Bilingual Word Embeddings from Non-Parallel Document-Aligned Data Applied to Bilingual Lexicon Induction* ([notes](./Vulic-Moens-2015.md)): Explicates the bilingual induction task against which Duong et al. test their embeddings.


### Things to follow up on (?)
* [Rfou et al](./Rfou-et-al-2013.md) reference "SENNA" embeddings - an earlier semi-supervised approach to learning phrases that is somehow efficient (avoids normalization?).
* [Irvine and Callison-Burch 2017](http://www.mitpressjournals.org/doi/full/10.1162/COLI_a_00284): This is an overview of the bilingual induction task from MIT's in house journal. It includes a history of early methods... maybe we could show the progression of performance from early methods to contemporary ones? -- I'd like to give this article a closer read. -- MV
* LM Thang's [Neural Machine Translation Thesis](https://github.com/lmthang/thesis/blob/master/thesis.pdf) See Chapters 1 & 2 for an overview/ history of NMT.
* Google NLP group's [Tensorflow NMT tutorial](https://research.googleblog.com/2017/07/building-your-own-neural-machine.html)
* Mikolov's [Noise Contrastive Estimation Paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) "Distributed Representation of Words and Phrases and their Compositionality"
* After reading the MIT article by [Irvine-Callison-Burch 2017](./Irvine-Callison-Burch-2017.md) I wonder if we should focus narrowly on a translation task that is 1) lay-person comprehensible (eg. polysemy, or use of idiom) and then demonstrate/discuss how the progression of the "state of the art" has shown better and better performance on this task (culminating in the Duong embeddings) <-- this would be a pretty major course shift... its probably not necessary, just potentially interesting.

###  New Research
* *Unsupervised Machine Translation
Using Monoligual Corpora Only* -- James directed us to this paper.  https://arxiv.org/pdf/1711.00043.pdf
* Idiom detection paper that Roseanna found -- http://www.aclweb.org/anthology/W16-1817

