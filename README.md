# EBM-NLP

This is the part of the class project for CSC 585 - Algorithms in NLP - at University of Arizona.

This is a multi-level named entity recognition problem. The dataset is described in the paper https://arxiv.org/abs/1806.04185 which contains multi-level annotations of patients, interventions and outcomes for medical literature.

There are two tasks for the two levels of annotations in the dataset. The first task is identification of P (patients), I (interventions), and O(outcomes) span. The baseline model implementation for this task uses a bidirectional LSTM with CRF tagger on top https://github.com/bepnye/EBM-NLP/ . 

The second task is token level detailed labeling of the PIO spans. This baseline hierarchical labeling is performed using stanford Named Entity Recognizer https://nlp.stanford.edu/software/CRF-NER.html

The models can be evaluated using docker. The embeddings and trained models are downloaded from box university account for evaluation.

## Nested Named Entity Recognition
Our approach is based on the paper "A neural layered model for nested named entity recognition" https://aclweb.org/anthology/N18-1131
The paper detects entities using a flat NER layer described in the paper from Lample etal. "Neural architectures for named entity recognition" https://arxiv.org/pdf/1603.01360.pdf starting from inner entities. On detecting the inner entities, they merge/average the contextual embedding for tokens in the inner entities, and feed them to another flat NER layer for outer entity detection. They stack the layers on top of each other until no new entities are detected.

## Our Approach


## Models Trained Using

OS: MacOS High Sierra v 10.13.6
2.9 GHz Intel Core i5
16 GB DDR3 RAM

## Docker Environment

OS: Ubuntu 16.04
Python 3.5

## Steps to run docker code

```
git clone https://github.com/devkotasabin/EBM-NLP.git
cd EBM-NLP
sudo docker build -t devkota-sabin-hw3 .
sudo docker run --name devkotasabin devkota-sabin-hw3
```

