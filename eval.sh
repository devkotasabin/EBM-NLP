#!/bin/sh
echo "Downloading word embeddings and classifier files"
echo ""
cd acl_scripts/lstm-crf/data/

wget --progress=bar:force:noscroll -O embeddings.200d.trimmed.npz -L https://arizona.box.com/shared/static/xr099y3v1m37kfcsk7jwmi5po4fkg8hq.npz -q --show-progress

cd /ebm-nlp/acl_scripts/lstm-crf/results/test/

wget --progress=bar:force:noscroll -O events.out.tfevents.1538279398.Sabins-MacBook-Pro.local -L https://arizona.box.com/shared/static/ibfgvct1tyhm0s1nuvdx0u5h7nwyh415.local -q --show-progress

cd /ebm-nlp/stanford-ner/picodata/classifiers/

wget --progress=bar:force:noscroll -O patients.crf.ser.gz -L https://arizona.box.com/shared/static/r722kzvmmp4230dt6k2h5utzgon6ngzb.gz -q --show-progress

wget --progress=bar:force:noscroll -O interventions.crf.ser.gz -L  -q --show-progress

wget --progress=bar:force:noscroll -O outcomes.crf.ser.gz -L https://arizona.box.com/shared/static/mssu94b9e6i66e6lsimw4ovjsim933l3.gz -q --show-progress

cd /ebm-nlp/acl_scripts/lstm-crf/

echo ""
echo "Evaluation of PIO span identification"

python3 evaluate.py

echo ""

cd /ebm-nlp/stanford-ner/

echo "Token level hierarchical labeling"
echo ""
echo "For patients"

java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier picodata/classifiers/patients.crf.ser.gz -outputFormat tsv -testFile picodata/patients/test.txt > test_patients.tsv

echo ""
echo "For interventions"

java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier picodata/classifiers/interventions.crf.ser.gz -outputFormat tsv -testFile picodata/interventions/test.txt > test_interventions.tsv

echo ""
echo "For outcomes"

java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier picodata/classifiers/outcomes.crf.ser.gz -outputFormat tsv -testFile picodata/outcomes/test.txt > test_outcomes.tsv

cd ..

