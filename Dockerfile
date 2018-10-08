# Setup container with Ubuntu 16.04 image
FROM ubuntu:16.04

# Work directory
# Set the working directory to /ebm-nlp
WORKDIR /ebm-nlp

# Copy the current directory contents into the container at /ebm-nlp
COPY . /ebm-nlp

# Update container image
RUN apt-get -qq update

RUN apt-get -qq install -y zip unzip wget git -qq
RUN apt-get -qq install -y python3 python3-pip -qq
RUN apt-get -qq install -y openjdk-8-jre-headless -qq

RUN python3 -m pip -q install --upgrade setuptools

# Install Tensorflow
RUN python3 -m pip -q install --user tensorflow

# Install NLTK library and Averaged Perceptron Tagger
RUN python3 -m pip -q install nltk
RUN python3 -m nltk.downloader averaged_perceptron_tagger

# Clean accessibility.properties for java
# RUN sed -i 's/assistive/#assistive/g' /etc/java-8-openjdk/accessibility.properties

# Add stanford-ner to CLASSPATH
RUN export CLASSPATH="/ebm-nlp/stanford-ner/stanford-ner.jar"

# Set executable permission to test models
RUN chmod +x runall.sh
RUN chmod +x eval.sh

CMD ["/bin/sh", "./eval.sh"]
