**Täckström et al. (2012): [Cross-lingual Word Clusters for Direct Transfer of Linguistic Structure](https://arxiv.org/pdf/1309.4168.pdf)**
* Applying a model built on a resource-rich language to a resource-poor language

**Klementiev (2012): [Inducing Crosslingual Distributed Representations of Words](http://www.aclweb.org/anthology/C12-1089)**
* Introducing a method to represent both source and target language (crosslingual word embeddings) as a dense vector in the same vector space
* Assigning similar embedding vectors to translations; method suffers from errors in automatic word alignment

**Mikolov et al. (2013): [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/pdf/1309.4168.pdf)**
* Introducing contextual bag of words model (CBOW): learning word representations based on distributional contexts
* Using a bilingual dictionary as a source of bilingual information (as opposed to identical bilingual texts)
* Filtering a bilingual dictionary for one-to-one translations sidesteps issues with noisy dictionaries, but discards information
* Using cascade-style training (embeddings in source and target language are learned separately and then combined), which appears less effective than joint training of embeddings

