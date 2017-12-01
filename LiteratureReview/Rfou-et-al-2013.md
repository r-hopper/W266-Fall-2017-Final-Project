### [Paper Introducing Polyglot](http://www.aclweb.org/anthology/W13-3520) (I think?)
**Summary**
* The authors describe the challenge with building multilingual processing systems (_what do they mean by this practicalys?_) as follows:
  *"These systems are typically carefully tuned with hand-manufactured features designed by experts in a particular language. This approach can yield good performance, but tends to create complicated systems which have limited portability to new languages, in addition to being hard to enhance and maintain."
* ... an suggest that unsupervised learning of word embeddings is a potential solution:
  * "Recent advancements in unsupervised feature learning present an intriguing alternative. In- stead of relying on expert knowledge, these approaches employ automatically generated task- independent features (or word embeddings) given large amounts of plain text. "
* This paper investigates embedding performance on the POS task.
* They reference "SENNA" embeddings -- an earlier approach (semi-supervised) which "distinguishes between phrases and corrupted versions of them. In doing this, the model avoids the need to normalize the scores across the vocabulary to infer probabilities. (Chen et al., 2013)"... in other words this approach (and the one used by these authors) is training word embeddings using a language modeling task (following Bengio et al 2009).

**Questions/Ideas**
* _Note how the "task-independence" is touted as one of the benefits of unsupervised word embeddings. They cite (Collobert and Weston, 2008) when talking about the use of word embeddings to replace "NLP common features"_
* _tangentially related: I'd like to develop a layperson's intuition for why multidimensional continuous representations for word meanings make sense given the realities of language use -- I think it would be interesting for me, important to justify whether this is a niche skill or a conceptual advancement, and useful for when we present our final project._ Note to self: is [Quine](https://en.wikipedia.org/wiki/Radical_translation) relevant here?

**Citation**
> Rami Al-Rfou, Bryan Perozzi and Steven Skiena. "Polyglot: Distributed Word Representations for Multilingual NLP" _Proceedings of the Seventeenth Conference on Computational Natural Language Learning_ pages 183–192, Sofia, Bulgaria, August 8-9 2013. ⃝c 2013 Association for Computational Linguistics
