## Thesis: Irony detection and polarity classification in Tripadvisor user reviews

In this repository you will find several models for the classification of Trip Advisor reviews by irony and polarity, using supervised learning algorithms.

The first dataset, Sarcasm V2 [https://nlds.soe.ucsc.edu/sarcasm2], was constructed from a large-scale and highly diverse dialogue corpus of online discussion forums to classify into sarcasm (S) and non-sarcasm (NS) classes, distributed into three balanced sample subsets based on generic sarcasm (Gen), rhetorical questions (RQ), and hyperbole (Hyp)(Oraby et al., 2017). 

The second dataset are 1254 reviews about Amazon productos [https://storm.cis.fordham.edu/~filatova/SarcasmCorpus.rar].

The third dataset consists of 20,490 hotel reviews extracted from TripAdvisor [https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews/download?datasetVersionNumber=2], which contain the star rating (up to 5) given by the user (Alam et al., 2016).

For word representation, three GloVe word embedding models are used, 
1. with 6 billion words and 100 dimensions words: https://nlp.stanford.edu/data/glove.6B.zip
2. with 6 billion words and 300 dimensions words: https://nlp.stanford.edu/data/glove.6B.zip
3. with 840 billion words and 300 dimensions words: https://nlp.stanford.edu/data/glove.840B.300d.zip
 
Download and unzip "glove.6B.100d.txt", "glove.6B.300d.txt" from https://nlp.stanford.edu/data/glove.6B.zip, and "glove.840B.300d.txt" from https://nlp.stanford.edu/data/glove.840B.300d.zip

Packages used (Python 3.7):
- Tensorflow 2.0
- Scikit-learn 1.0.2
- Pandas 1.3.5
- Numpy 1.21.5
- h5py 2.10.0

### References:
- Alam, M. H., Ryu, W.-J., & Lee, S. (2016). Joint multi-grain topic sentiment: Modeling semantic aspects for online reviews. Information Sciences, 339, 206-223.
- Filatova, E. (2012). Irony and Sarcasm: Corpus Generation and Analysis Using Crowdsourcing. Lrec, 392-398.
- Oraby, S., Harrison, V., Reed, L., Hernandez, E., Riloff, E., & Walker, M. (2017). Creating and characterizing a diverse corpus of sarcasm in dialogue. arXiv preprint arXiv:1709.05404.
