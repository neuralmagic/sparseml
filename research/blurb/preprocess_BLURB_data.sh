#!/bin/sh

# For Chemprot corpus, please register and download from https://biocreative.bioinformatics.udel.edu/resources/corpora/chemprot-corpus-biocreative-vi/
if [ -s raw_data/ChemProt_Corpus.zip ]
then 
    unzip raw_data/ChemProt_Corpus.zip -d raw_data/
    unzip raw_data/ChemProt_Corpus/chemprot_training.zip -d raw_data/ChemProt_Corpus/chemprot_training
    unzip raw_data/ChemProt_Corpus/chemprot_development.zip -d raw_data/ChemProt_Corpus/chemprot_development
    unzip raw_data/ChemProt_Corpus/chemprot_test_gs.zip -d raw_data/ChemProt_Corpus/chemprot_test_gs
fi
# For BioASQ corpus,  please register and download BioASQ7 (both train and test) from http://participants-area.bioasq.org/datasets/
if [ -s raw_data/BioASQ-training7b.zip ]
then
    mkdir raw_data/BioASQ
    unzip raw_data/BioASQ-training7b.zip -d raw_data/BioASQ/
    unzip raw_data/Task7BGoldenEnriched.zip -d raw_data/BioASQ/
fi

python preprocessor.py 
cd raw_data/pubmedqa/preprocess/
python split_dataset.py pqal
cd ../../..
cp -r raw_data/pubmedqa/data data/
mv data/data data/pubmedqa 