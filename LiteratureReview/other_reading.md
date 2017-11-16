# Maya's Reading Notes
I need to figure out the right place in the repo for this stuff... just putting it here for now...

### [Bablyon Partners Repo](https://github.com/Babylonpartners/fastText_multilingual), Readme & [associated paper](https://arxiv.org/abs/1702.03859).

**Summary**
* Mikolov et. al showed (_in which paper ?_) that bilingual word vectors can be trained "offline" (_i.e. on monolingual corpora_) and then aligned using a vector transformation that preserves the similarity measures within each language respectively.
* Mikolov used a bilingual dictionary to learn this^^ transformation. These authors introduce a "novel inverted softmax" for identifying translation pairs & are able to improve on their
* This paper proves that the linear transformation should be orthogonal & that it can be obtained using singular value decomposition.
* The repo links to pretrained embeddings released by Facebook and provides a set of matrix transformations that can be used to align language.
* The authors imply that they are able to use a transformation to align languages for which they didn't have a bilingual signal at all: "Orthogonal transformations are more robust to noise, enabling us to learn the transformation without expert bilingual signal by constructing a "pseudo-dictionary" from the identical character strings which appear in both languages, achieving 40% precision on the same test set. "

**Questions/Ideas**  
* _could these vectors be useful to us as a point of comparison to the ones we train ourself?_
* _this work is clearly more recent than the Duong paper we are looking at, is it building on a similar idea? how exactly are these embeddings being trained, aligned etc.?_
* _the idea of being able to translate between languages without a bilingual signal seems to be related to transfer learning? If we could do that in an intelligible way what could that model tell us about our assumptions of language (where is the similarity coming from?)_
* _They use nearest neighbors to predict translations from their word embeddings -- is this what Duong refers to as the 'bilingual induction task' (I really need to look into that!)_

**citation**
> Offline bilingual word vectors, orthogonal transformations and the inverted softmax
Samuel L. Smith, David H. P. Turban, Steven Hamblin and Nils Y. Hammerla
ICLR 2017 (conference track)


### [Paper Introducing Polyglot](http://www.aclweb.org/anthology/W13-3520) (I think?)
**Summary**
* The authors describe the challenge with building multilingual processing systems (_what do they mean by this practicallys?_) as follows:
  *"These systems are typically carefully tuned with hand-manufactured features designed by experts in a particular language. This approach can yield good performance, but tends to create complicated systems which have limited portabil- ity to new languages, in addition to being hard to enhance and maintain."
* ... an suggest that unsupervised learning of word embeddings is a potential solution:
  * "Recent advancements in unsupervised feature learning present an intriguing alternative. In- stead of relying on expert knowledge, these ap- proaches employ automatically generated task- independent features (or word embeddings) given large amounts of plain text. "
* This paper investigates embedding performance on the POS task.
* They reference "SENNA" embeddings -- an earlier approach (semi-supervised) which "distinguishes between phrases and corrupted versions of them. In doing this, the model avoids the need to normalize the scores across the vocab- ulary to infer probabilities. (Chen et al., 2013)"... in other words this approach (and the one used by these authors) is training word embeddings using a language modeling task (following Bengio et al 2009).

**Questions/Ideas**
* _Note how the "task-independence" is touted as one of the benefits of unsupervised word embeddings. They cite (Collobert and Weston, 2008) when talking about the use of word embeddings to replace "NLP common features"_ 
* _tangentially related: I'd like to develop a layperson's intuition for why multidimensional continuous representations for word meanings make sense given the realities of language use -- I think it would be interesting for me, important to justify whether this is a niche skill or a conceptual advancement, and useful for when we present our final project._ Note to self: is [Quine](https://en.wikipedia.org/wiki/Radical_translation) relevant here?

**Citation**
> Rami Al-Rfou, Bryan Perozzi and Steven Skiena. "Polyglot: Distributed Word Representations for Multilingual NLP" _Proceedings of the Seventeenth Conference on Computational Natural Language Learning_ pages 183–192, Sofia, Bulgaria, August 8-9 2013. ⃝c 2013 Association for Computational Linguistics

