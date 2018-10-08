glove:
	wget -P ./data/ "http://nlp.stanford.edu/data/glove.6B.zip"
	unzip ./data/glove.6B.zip -d data/glove.6B/
	rm ./data/glove.6B.zip

pubmed:
	wget -P ./data/embeddings/ "https://arizona.box.com/shared/static/8npnjhn3zulr6hspy6ei71mq92qkmjwf.txt"

run:
	python3 build_data.py
	python3 train.py
	python3 evaluate.py
