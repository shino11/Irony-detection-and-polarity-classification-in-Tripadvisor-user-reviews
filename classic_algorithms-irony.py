import gensim.downloader as api
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from gensim.models import KeyedVectors
from sklearn.tree import DecisionTreeClassifier

from Word2VecVectorizer import Word2VecVectorizer
import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


X = []
Y = []
Label_encoder = LabelEncoder()
df = pd.read_csv('C:/Users/ShinoSan/Jupyter/sarcasm_v2(0-1).csv', encoding='ISO-8859-1', sep=',')
df2 = pd.read_csv('C:/Users/ShinoSan/Jupyter/Tripadvisor_hotel_reviews-rating-sentiments-PossibleIrony.csv', encoding='ISO-8859-1', sep=',')


# Train models on Sarcasm dataset
def read_sarcasm_dataset():
    cols = ['Corpus', 'Label', 'ID', 'Quote Text', 'Response Text']
    df.columns = cols
    # global Label
    x1 = df.loc[:, 'Response Text']
    a1 = df.loc[:, 'Quote Text']
    label = df.loc[:, 'Label']
    print(df.shape)
    x_featured = []
    for i in range(x1.count()):
        x_featured.append(x1[i] + ' ' + a1[i])
    x_featured = np.reshape(x_featured, x1.shape)
    print(x_featured.shape)
    word_embedding(x_featured)
    # Label
    global Y
    Y = Label_encoder.fit_transform(label)
    print(Y.shape)


# # classify tripadvisor dataset
def read_tripadvisor_dataset():
    cols = ['Review', 'Rating', 'Sentiment_Score', 'Sentiment', 'Possible_Irony']
    df.columns = cols
    x2 = df2.loc[:, 'Review']
    a = df2.loc[:, 'Rating']
    b = df2.loc[:, 'Sentiment_Score']
    c = df2.loc[:, 'Sentiment']
    d = df2.loc[:, 'Possible_Irony']
    print(df2.shape)
    word_embedding(x2)


# glove with format of word2vec
def word_embedding (x):
    global X
    model = KeyedVectors.load_word2vec_format('glove.word2vec', binary=False)
    # model = KeyedVectors.load_word2vec_format('fasttext..fasttext-wiki-news-subwords-300', binary=False) no
    # model = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data   fasttext-wiki-news-subwords-300 no
    # model = api.load("fasttext-wiki-news-subwords-300")  # load pre-trained word-vectors from gensim-data ok
    vectorizer = Word2VecVectorizer(model)
    # Get the sentence embeddings for the train dataset
    X = vectorizer.fit_transform(x)  # X_featured
    print(X.shape)


def distribution(d, name, col, tag):
    c = d.groupby([tag])[col].count()
    # print(d)
    dff = pd.DataFrame.from_dict(c)
    dff = dff.transpose()
    dff.to_csv(name, encoding='ISO-8859-1', mode='a')


# def getEmbedding(EMBEDDING_DIM):
#     embeddings_index = {}
#     count = 0
#     words = []
#     f = open('data/word_embeddings/glove.840B.300d.txt', encoding='utf-8', errors='ignore')
#     # f = open('data/word_embeddings/glove.6B.100d.txt', encoding='utf-8', errors='ignore')
#     for line in f:
#         values = line.split()
#         word = values[0]
#         words.append(word)
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#         count = count + 1;
#     f.close()
#
#     tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#     tokenizer.fit_on_texts(words)
#     word_index = tokenizer.word_index
#
#     print("total words embeddings is ", count, len(word_index))
#     embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#     for word, i in word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#
#     # embedding_layer = Embedding(input_dim=len(word_index) + 1,
#     #                             output_dim=EMBEDDING_DIM,
#     #                             weights=[embedding_matrix],
#     #                             input_length=MAX_SEQUENCE_LENGTH,
#     #                             trainable=True)
#     return tokenizer, embedding_matrix


#
# def tf_idf(train, test):
#     # corpus = ['This is the first document.','This document is the second document.', 'And this is the third one.', 'Is this the first document?' ]
#     tfidf_vectorizer = TfidfVectorizer()
#     # X = tfidf_vectorizer.fit_transform(corpus)
#     # tfidf_vectorizer.get_feature_names_out()
#     # print(X.shape)
#     tfidf_train_vectors = tfidf_vectorizer.fit_transform(train)
#     tfidf_test_vectors = tfidf_vectorizer.transform(test)
#     return tfidf_train_vectors.toarray(), tfidf_test_vectors.toarray()
#
#
# # X, Y = tf_idf(X, Label)  NO
#
#
# def naive_bayes():
#     precision_list = list()
#     recall_list = list()
#     accuracy_list = list()
#     f1_list = list()
#     classifier = GaussianNB()
#     # classifier = DecisionTreeClassifier(random_state=0)
#     # print(Label)
#
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#     for train, test in kfold.split(X, Y):
#         # print(X[train].shape, Y[train].shape)
#         # x_train,x_test,y_train,y_test = train_test_split(X,Label.iloc[:,1],test_size=0.8, random_state=0, stratify=Label.iloc[:,1])
#         # print(Label[train])
#         classifier.fit(X[train], Y[train])
#         labels_pred = classifier.predict(X[test])
#         # tfidfXtrain, tfidfXtest = tf_idf(X[train], X[test])
#         # classifier.fit(tfidfXtrain, Y[train])
#         # labels_pred = classifier.predict(tfidfXtest)
#         # print(labels_pred)
#         # print(Y[test])
#         b_acc = balanced_accuracy_score(Y[test], labels_pred)
#         f1 = f1_score(Y[test], labels_pred, average='weighted')
#         precision = precision_score(Y[test], labels_pred, average='weighted')
#         recall = recall_score(Y[test], labels_pred, average='weighted')
#         # print('Test b_acc:', b_acc)
#         # print('Test precision:', precision)
#         # print('Test recall:', recall)
#         # print('Test f1:', f1)
#         accuracy_list.append(b_acc)
#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)
#
#     print('avg b_accuracy precision recall f1')
#     # print('avg b_accuracy {0}'.format(np.average(accuracy_list)))
#     # print('avg precision {0}'.format(np.average(precision_list)))
#     # print('avg recall {0}'.format(np.average(recall_list)))
#     # print('avg f1 {0}'.format(np.average(f1_list)))
#     print(np.average(accuracy_list))
#     print(np.average(precision_list))
#     print(np.average(recall_list))
#     print(np.average(f1_list))
#     plt.figure()
#     plt.plot(f1_list, color='darkorange', lw=2, label='F1')
#     plt.show()
#
#
# def decision_tree():
#     precision_list = list()
#     recall_list = list()
#     accuracy_list = list()
#     f1_list = list()
#     classifier = DecisionTreeClassifier(random_state=0)
#
#     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
#     for train, test in kfold.split(X, Y):
#         # print(X[train].shape, Y[train].shape)
#         # x_train,x_test,y_train,y_test = train_test_split(X,Label.iloc[:,1],test_size=0.8, random_state=0, stratify=Label.iloc[:,1])
#         # print(Label[train])
#         classifier.fit(X[train], Y[train])
#         labels_pred = classifier.predict(X[test])
#         # tfidfXtrain, tfidfXtest = tf_idf(X[train], X[test])
#         # classifier.fit(tfidfXtrain, Y[train])
#         # labels_pred = classifier.predict(tfidfXtest)
#         # print(labels_pred)
#         # print(Y[test])
#         b_acc = balanced_accuracy_score(Y[test], labels_pred)
#         f1 = f1_score(Y[test], labels_pred, average='weighted')
#         precision = precision_score(Y[test], labels_pred, average='weighted')
#         recall = recall_score(Y[test], labels_pred, average='weighted')
#         # print('Test b_acc:', b_acc)
#         # print('Test precision:', precision)
#         # print('Test recall:', recall)
#         # print('Test f1:', f1)
#         accuracy_list.append(b_acc)
#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)
#
#     print('avg b_accuracy precision recall f1')
#     # print('avg b_accuracy {0}'.format(np.average(accuracy_list)))
#     # print('avg precision {0}'.format(np.average(precision_list)))
#     # print('avg recall {0}'.format(np.average(recall_list)))
#     # print('avg f1 {0}'.format(np.average(f1_list)))
#     print(np.average(accuracy_list))
#     print(np.average(precision_list))
#     print(np.average(recall_list))
#     print(np.average(f1_list))
#     plt.figure()
#     plt.plot(f1_list, color='darkorange', lw=2, label='F1')
#     plt.show()


def randon_forest_exp():
    global X, Y
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = RandomForestClassifier(random_state=42, verbose=1, n_estimators=100, n_jobs=4)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        # print(X[train].shape, Y[train].shape)
        classifier.fit(X[train], Y[train])
        # acc = classifier.score(X[test], Y[test])
        # print("Accuracy: {0:.2%}".format(acc))
        labels_pred = classifier.predict(X[test])
        # print(accuracy_score(Y[test], labels_pred))
        b_acc = accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def KNN_exp():
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = KNeighborsClassifier(weights='distance', n_neighbors=10)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        classifier.fit(X[train], Y[train])
        labels_pred = classifier.predict(X[test])
        b_acc = accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def SVM_exp():
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = SVC(gamma='auto')
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        classifier.fit(X[train], Y[train])
        labels_pred = classifier.predict(X[test])
        b_acc = accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def naive_bayes_exp():
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = GaussianNB()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        classifier.fit(X[train], Y[train])
        labels_pred = classifier.predict(X[test])
        b_acc = accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def logistic_regression_exp():
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = LogisticRegression(random_state=42)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        classifier.fit(X[train], Y[train])
        labels_pred = classifier.predict(X[test])
        b_acc = accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def randon_forest_model_training():
    classifier = RandomForestClassifier(random_state=42, verbose=1, n_estimators=100, n_jobs=4)
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, "rf.pkl")  # save
    model = joblib.load("rf.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def KNN_model_training():
    classifier = KNeighborsClassifier(weights='distance', n_neighbors=10)
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    classifier.fit(x_train, y_train)
    import joblib
    joblib.dump(classifier, "knn.pkl")  # save
    model = joblib.load("knn.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def SVM_model_training():
    classifier = SVC(gamma='auto')
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    classifier.fit(x_train, y_train)
    import joblib
    joblib.dump(classifier, "svm.pkl")  # save
    model = joblib.load("svm.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def naive_bayes_model_training():
    classifier = GaussianNB()
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, "nb.pkl")  # save
    model = joblib.load("nb.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def logistic_regression_model_training():
    classifier = LogisticRegression(random_state=42)
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, "lr.pkl")  # save
    model = joblib.load("lr.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def randon_forest_classifier():
    model = joblib.load("rf.pkl")  # load
    labels_pred = model.predict(X) # classify new
    new_labels = Label_encoder.inverse_transform(labels_pred)
    print(new_labels)
    # for i in range(len(labels_pred)):
    #     print(X2[i], new_labels[i])


# def SVM_classifier():
#     model = joblib.load("svm.pkl")  # load
#     labels_pred = model.predict(X)  # classify new
#     new_labels = Label_encoder.inverse_transform(labels_pred)
#     print(new_labels)


# def KNN_clasifier():
#     model = joblib.load("knn.pkl")  # load
#     labels_pred = model.predict(X) # classify new
#     new_labels = Label_encoder.inverse_transform(labels_pred)
#     print(new_labels)
#     # for i in range(len(labels_pred)):
#     # print(X[i], new_labels[i])
#     # print(labels_pred.inverse_transform())


def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    param_range = [x[1] for x in param_range]
    sort_idx = np.argsort(param_range)
    param_range=np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Weight of class 2')
    plt.ylabel('Average values and standard deviation for F1-Score')
    plt.legend(loc='best')
    plt.show()


read_sarcasm_dataset()
# irony experiment
print("EXPERIMENTS CROSS-VALIDATION FOLD=10---------------------------------")
print("RF---------------------------------")
randon_forest_exp()
print("KNN---------------------------------")
KNN_exp()
print("SVM---------------------------------")
SVM_exp()
print("NB---------------------------------")
naive_bayes_exp()
print("LR---------------------------------")
logistic_regression_exp()
# training and saving models
print("TRAINING 90-10---------------------------------")
print("RF---------------------------------")
randon_forest_model_training()
print("KNN---------------------------------")
KNN_model_training()
print("SVM---------------------------------")
SVM_model_training()
print("NB---------------------------------")
naive_bayes_model_training()
print("LR---------------------------------")
logistic_regression_model_training()

# classifying
# read_tripadvisor_dataset()
# print("RF---------------------------------")
# randon_forest_classifier()

# distribution(df2, 'distrib_tripadvisor_irony.csv', 'Review', '')