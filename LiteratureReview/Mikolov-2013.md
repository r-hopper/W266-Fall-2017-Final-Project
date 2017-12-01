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
