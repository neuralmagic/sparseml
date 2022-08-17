#!/bin/sh

# Downloading NER data.
mkdir raw_data
mkdir data
git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016.git raw_data/MTL-Bioinformatics-2016
mkdir raw_data/JNLPBA
wget -O raw_data/JNLPBA/Genia4ERtraining.tar.gz http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz
wget -O raw_data/JNLPBA/Genia4ERtest.tar.gz http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz
tar zxvf raw_data/JNLPBA/Genia4ERtraining.tar.gz -C raw_data/JNLPBA/
tar zxvf raw_data/JNLPBA/Genia4ERtest.tar.gz -C raw_data/JNLPBA/

# Downloading EBM-NLP PICO corpus.
git clone https://github.com/bepnye/EBM-NLP.git raw_data/EBM-NLP
tar -xzf raw_data/EBM-NLP/ebm_nlp_2_00.tar.gz -C raw_data/EBM-NLP

# Downloading DDI corpus.
git clone https://github.com/zhangyijia1979/hierarchical-RNNs-model-for-DDI-extraction.git raw_data/DDI
tar -zxvf raw_data/DDI/DDIextraction2013/DDIextraction_2013.tar.gz -C raw_data/DDI/

# Downloading BIOSSES corpus.
wget -O raw_data/BIOSSES-Dataset.rar https://tabilab.cmpe.boun.edu.tr/BIOSSES/Downloads/BIOSSES-Dataset.rar
unrar x raw_data/BIOSSES-Dataset.rar raw_data/

# Downloading PubmedQA corpus.
git clone https://github.com/pubmedqa/pubmedqa.git raw_data/pubmedqa

# Downloading HoC corpus
git clone https://github.com/sb895/Hallmarks-of-Cancer.git raw_data/hoc

#Downloading GAD corpus
wget -O raw_data/REdata.zip https://drive.google.com/u/0/uc?id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw
unzip raw_data/REdata.zip -d raw_data/