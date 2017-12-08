**Vulić and Moens (2013): [Cross-Lingual Semantic Similarity of Words as the Similarity of Their Semantic Word Responses](http://www.aclweb.org/anthology/N13-1011)**
* Describes their methodology for choosing 1000 words for "ground-truth translation" for the bilingual induction task
  * "Since our task is bilingual lexicon extraction, we designed a set of ground truth one-to-one translation pairs for all 3 language pairs as follows. 
    * For Dutch-English and Spanish-English, we randomly sampled a set of Dutch (Spanish) nouns from our Wikipedia corpora.
    * Following that, we used the Google Translate tool plus an additional annotator to translate those words to English. 
      * The annotator manually revised the lists and retained only words that have their corresponding translation in the English vocabulary.
      * Additionally, only one possible translation was annotated as correct. When more than 1 translation is possible, the annotator marked as correct the translation that occurs more frequently in the English Wikipedia data. Finally, we built a set of 1000 one-to-one translation pairs for Dutch-English and Spanish-English. The same procedure was followed for Italian-English, but there we obtained the ground truth one-to-one translation pairs for 1000 most frequent Italian nouns in order to test the effect of word frequency on the quality of semantic word responses and the overall lexicon quality."
      
      
* Describes how they assess accuracy
  * "All the methods under consideration actually retrieve ranked lists of semantically similar words that could be observed as potential translation candidates. 
  * We measure the performance as Top M accuracy ($Acc_{M}$). It denotes the number of source words from ground truth translation pairs whose top M semantically similar words contain the correct translation according to our ground truth over the total number of ground
truth translation pairs (=1000).
      
      
      
*Notes:
* The list of 1000 words is supposedly available online but I'm unable to access the webpage: [http://people.cs.kuleuven.be/∼ivan.vulic/software/](http://people.cs.kuleuven.be/∼ivan.vulic/software/)
* I'm emailing Vulić directly about the resource
