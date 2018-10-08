#!/bin/sh
tar -xvzf ebm_nlp_1_00.tar.gz
cd acl_scripts/lstm-crf/
make pubmed
make run
cd ..
cd ..