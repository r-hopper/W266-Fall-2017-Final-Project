**Levy and Goldberg (2014): [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization)**
* Intro states that skip-gram with negative sampling (SGNS), as proposed by Mikolov et al. is "highly popular but not well understood"
* Proposes that SGNS's training method is weighted matrix factorization, and that its objective is factorizing a "shifted PMI matrix" (a PMI matrix shifted by a constant offset)
  * Proposes that SGNS performs weighted matrix factorization, which weights frequent pairs more highly, which excels on word-analogy tasks
* Consider a word-context pair (w,c). Did this pair come from the observed data D?
  * "The negative sampling objective tries to maximize $P(D=1 | w,c)$ for observed (w,c) pairs while maximizing $P(D=0 | w,c)$ for randomly sampled "negative" examples, under the assumption that randomly selecting a context for a given word is likely to result in an unobserved (w,c) pair.
* SGNS results in word and context matrices W and C. Consider the product W * C^T = M.
  * SGNS is factorizing a matrix in which each row corresponds to a word and each column corresponds to a context, and each cell contains a quantity f(w,c) reflecting the strength of association between that particular word-context pair.
* What can we say about f(w,c)?
  * For a negative-sampling value of k=1, the SGNS objective is factorizing a word-context matrix in which the association between a word and its context is given by f(w,c)=PMI(w,c).
  
* Paper also discusses the issues that lead to shifting between sparse/non-sparse matrices and how to represent observed word-context pairs with low frequency (negative matrix value), high frequency (positive value), and unobserved pairs (0 value)


Notes to self:
* Explore how NCE is specifically different from negative sampling as described by Mikolov et al.
