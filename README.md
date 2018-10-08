# EBM-NLP

This is the part of the class project for CSC 585 - Algorithms in NLP - at University of Arizona.

This is a multi-level named entity recognition problem. The dataset is described in the paper https://arxiv.org/abs/1806.04185 which contains multi-level annotations of patients, interventions and outcomes for medical literature.

There are two tasks for the two levels of annotations in the dataset. The first task is identification of P (patients), I (interventions), and O(outcomes) span. The baseline model implementation for this task uses a bidirectional LSTM with CRF tagger on top https://github.com/bepnye/EBM-NLP/ . 

The second task is token level detailed labeling of the PIO spans. This hierarchical labeling is performed using stanford Named Entity Recognizer https://nlp.stanford.edu/software/CRF-NER.html

The models can be evaluated using docker. The embeddings and trained models are downloaded from box university account for evaluation.

## Models Trained Using

OS: MacOS High Sierra v 10.13.6
2.9 GHz Intel Core i5
16 GB DDR3 RAM

## Docker Environment

OS: Ubuntu 16.04


