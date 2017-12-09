#!/bin/bash

# Load MUSE ground truth translations from AWS
# Sourced from https://github.com/facebookresearch/MUSE 

##############################

# Get ?-en translations
wget "https://s3.amazonaws.com/arrival/dictionaries/es-en.txt" -O es-en.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/fr-en.txt" -O fr-en.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/it-en.txt" -O it-en.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/ja-en.txt" -O ja-en.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/nl-en.txt" -O nl-en.txt

# Get en-? translations
wget "https://s3.amazonaws.com/arrival/dictionaries/en-es.txt" -O en-es.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/en-fr.txt" -O en-fr.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/en-it.txt" -O en-it.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/en-ja.txt" -O en-ja.txt
wget "https://s3.amazonaws.com/arrival/dictionaries/en-nl.txt" -O en-nl.txt

# Exit
exit
