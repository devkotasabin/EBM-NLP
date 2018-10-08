#!/bin/sh
echo "Training classifiers for hierarchical labeling: "

java -mx10g -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop patient_train.prop

java -mx10g -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop intervention_train.prop

java -mx10g -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop outcome_train.prop

java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier picodata/classifiers/patients.crf.ser.gz -outputFormat tsv -testFile picodata/patients/test.txt > test_patients.tsv
java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier picodata/classifiers/interventions.crf.ser.gz -outputFormat tsv -testFile picodata/interventions/test.txt > test_interventions.tsv
java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier picodata/classifiers/outcomes.crf.ser.gz -outputFormat tsv -testFile picodata/outcomes/test.txt > test_outcomes.tsv