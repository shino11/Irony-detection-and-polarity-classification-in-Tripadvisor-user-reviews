# Irony detection and polarity classification in Tripadvisor user reviews.

 In this repository you will find several models for the classification of Trip Advisor reviews using  learning supervised algorithms.

In this repository you will find several models for the classification of dialogue acts with and without context. The models were trained with Schema-Guided Dialogue (SGD) data sets and have given acceptable results. The hyperparameters it has are the ones that have offered the best results during the training process.

The first dataset, Sarcasm V2 , was constructed from a large-scale and highly diverse dialogue corpus of online discussion forums to classify into sarcasm (S) and non-sarcasm (NS) classes, distributed into three balanced sample subsets based on generic sarcasm (Gen), rhetorical questions (RQ), and hyperbole (Hyp). 

The dataset used is from the 8th Dialog System Technology Challenge 2019 (https://github.com/google-research-datasets/dstc8-schema-guided-dialogue). These were converted to a multiple csv files.

For word representation, Is used three GloVe word embedding models, 
1. with 6 billion words and 100 dimensions words: (https://nlp.stanford.edu/data/glove.6B.zip)
2. with 6 billion words and 300 dimensions words: (https://nlp.stanford.edu/data/glove.6B.zip)
3. with 840 billion words and 300 dimensions words: (https://nlp.stanford.edu/data/glove.840B.300d.zip)
 

Download and unzip "glove.6B.100d.txt", "glove.6B.300d.txt" from https://nlp.stanford.edu/data/glove.6B.zip, and "glove.840B.300d.txt" from https://nlp.stanford.edu/data/glove.840B.300d.zip

Packages used (Python 3.7):
- Tensorflow 2.0
- Scikit-learn 1.0.2
- Pandas 1.3.5
- Numpy 1.21.5
- h5py 2.10.0
