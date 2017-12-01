# Final Project: Learning Crosslingual Word Embeddings
>   
_Roseanna Hopper, Mona Iwamoto, Maya Miller-Vedam_   
w266 : Natural Language Processing, Fall 2017    
UC Berkeley School of Information, MIDS Program  

**About this Project**
_fill in smth. here before sharing this repo with others_


## Quick Links
*  [Original Paper](https://arxiv.org/pdf/1606.09403.pdf) Duong, Long & Kanayama, Hiroshi & Ma, Tengfei & Bird, Steven & Cohn, Trevor. (2016). Learning Crosslingual Word Embeddings without Bilingual Corpora. 1285-1295. 10.18653/v1/D16-1136.
* [PanLex](https://panlex.org/) ; plus their [API Documentation](https://dev.panlex.org/api/)
* [Polyglot](http://polyglot.readthedocs.io/en/latest/modules.html) ; plus Rami Al-Rfou's [Demo & Tutorial](https://sites.google.com/site/rmyeid/projects/polyglot) including [Wikipedia Text Dumps](https://sites.google.com/site/rmyeid/projects/polyglot#TOC-Download-Wikipedia-Text-Dumps)

## TO DO
* Work through [this Tensorflow NMT tutorial](https://research.googleblog.com/2017/07/building-your-own-neural-machine.html) (all)
* Load Raw Data into G Cloud Instance (Maya)
>> UPDATE 11/30: Done. See `/home/mmillervedam/Data` on the shared instance.
* Get C code running (Mona)
> UPDATE 11/28: Compiled & running but direct conversion to Python does not seem like a good approach. Alternative options are to run the C code in python using a wrapper like in [this Word2Vec implementation](https://github.com/danielfrg/word2vec/blob/master/setup.py) or rebuild from scratch following Word2Vec code itself.
* Make a G-cloud instance we can all access (Maya)
> UPDATE 11/30: This is all set, I just need Mona's public key. You should be able to ssh in using `gcloud compute --project "synthetic-arc-178620" ssh --zone "us-west1-b" "finalproject" --ssh-flag "-L 8888:127.0.0.1:8888" --ssh-flag  "-L 6006:127.0.0.1:6006"`
>> NEXT STEPS: Just need one of you to confirm that this works!
* Literature Review about Crosslingual Word Embeddings(all)
  > ONGOING: see [Literature Review](./LiteratureReview) folder above.
* Draft Milestone due Nov 16 (Maya)
  > DONE 11/12: see [Deliverables](./Deliverables) folder above.
* Look for baseline model implementations in Python (Roseanna)
  > DONE 11/13: see [Baseline Models](./BaselineModels); they all appear to be in C.


## Other Links
* Open NMT [project website](http://opennmt.net/FAQ/#where-can-i-get-training-data-for-translation-from-x-to-x)
* OPUS [open source parallel corpus](http://opus.nlpl.eu) -- note that this cite also has a bunch of Machine Translation links.
* [Project Proposal G-Doc](https://docs.google.com/document/d/1KetDWpYzTtGK18eEqWcLiTbOCSH0ReBHH3oOpfCfLaI/edit)
