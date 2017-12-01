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
