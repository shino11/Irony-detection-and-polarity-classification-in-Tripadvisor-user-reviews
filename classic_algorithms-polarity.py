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
# df = pd.read_csv('sarcasm_v2(0-1).csv', encoding='ISO-8859-1', sep=',')
df = pd.read_csv('Sarcasm_Corpus_Amazon_Polarity_Corpus(ironic^non-ironic)', encoding='ISO-8859-1', sep='|')
df2 = pd.read_csv('Tripadvisor_hotel_reviews-Rating-Polarity-PossibleIrony.csv', encoding='ISO-8859-1', sep=',')


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


def read_irony_dataset_Amazon():
    cols = ['stars' , 'title' , 'date','author' , 'product','review' , 'irony']
    df.columns = cols
    # global Label
    x1 = df.loc[:, 'review']
    label = df.loc[:, 'irony']
    print(df.shape)
    word_embedding(x1)
    # Label
    global Y
    Y = Label_encoder.fit_transform(label)
    print(Y.shape)
    global param_range


# # classify tripadvisor dataset
def read_tripadvisor_dataset():
    cols = ['Review', 'Rating', 'Sentiment_Score', 'Polarity', 'Possible_Irony']
    df2.columns = cols
    x2 = df2.loc[:, 'Review']
    a = df2.loc[:, 'Rating']
    b = df2.loc[:, 'Sentiment_Score']
    label = df2.loc[:, 'Polarity']
    d = df2.loc[:, 'Possible_Irony']
    print(df2.shape)
    word_embedding(x2)
    global Y
    Y = Label_encoder.fit_transform(label)
    print(Y.shape)


# glove with format of word2vec
def word_embedding (x):
    global X
    model = KeyedVectors.load_word2vec_format('glove.word2vec', binary=False)
    # model = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
    # model = api.load("fasttext-wiki-news-subwords-300")  # load pre-trained word-vectors from gensim-data fasttext-wiki-news-subwords-300
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


def random_forest_exp():
    global X, Y
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=4)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        # print(X[train].shape, Y[train].shape)
        classifier.fit(X[train], Y[train])
        # acc = classifier.score(X[test], Y[test])
        # print("Accuracy: {0:.2%}".format(acc))
        labels_pred = classifier.predict(X[test])
        # print(accuracy_score(Y[test], labels_pred))
        b_acc = balanced_accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg b_accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.title('random forest')
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
        b_acc = balanced_accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg b_accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.title('KNN')
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
        b_acc = balanced_accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg b_accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.title('SVM')
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
        b_acc = balanced_accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg b_accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.title('naive bayes')
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def logistic_regression_exp():
    precision_list = list()
    recall_list = list()
    accuracy_list = list()
    f1_list = list()
    classifier = LogisticRegression(random_state=42, max_iter=300, n_jobs=4)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        classifier.fit(X[train], Y[train])
        labels_pred = classifier.predict(X[test])
        b_acc = balanced_accuracy_score(Y[test], labels_pred)
        f1 = f1_score(Y[test], labels_pred, average='weighted')
        precision = precision_score(Y[test], labels_pred, average='weighted')
        recall = recall_score(Y[test], labels_pred, average='weighted')
        accuracy_list.append(b_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    print('avg b_accuracy precision recall f1')
    print(np.average(accuracy_list))
    print(np.average(precision_list))
    print(np.average(recall_list))
    print(np.average(f1_list))
    plt.figure()
    plt.title('logistic regression')
    plt.plot(f1_list, color='darkorange', lw=2, label='F1')
    plt.show()


def random_forest_model_training():
    classifier = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=4)
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, "rf_polarity.pkl")  # save
    model = joblib.load("rf_polarity.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = balanced_accuracy_score(y_test, labels_pred)
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
    joblib.dump(classifier, "knn_polarity.pkl")  # save
    model = joblib.load("knn_polarity.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = balanced_accuracy_score(y_test, labels_pred)
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
    joblib.dump(classifier, "svm_polarity.pkl")  # save
    model = joblib.load("svm_polarity.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = balanced_accuracy_score(y_test, labels_pred)
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
    joblib.dump(classifier, "nb_polarity.pkl")  # save
    model = joblib.load("nb_polarity.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = balanced_accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def logistic_regression_model_training():
    classifier = LogisticRegression(random_state=42, max_iter=300, n_jobs=4)
    ## Train model to save
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, "lr_polarity.pkl")  # save
    model = joblib.load("lr_polarity.pkl")  # load
    labels_pred = model.predict(x_test)
    b_acc = balanced_accuracy_score(y_test, labels_pred)
    f1 = f1_score(y_test, labels_pred, average='weighted')
    precision = precision_score(y_test, labels_pred, average='weighted')
    recall = recall_score(y_test, labels_pred, average='weighted')
    print(b_acc)
    print(precision)
    print(recall)
    print(f1)


def random_forest_classifier():
    model = joblib.load("rf_polarity.pkl")  # load
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
#     model = joblib.load("knn_irony.pkl")  # load
#     labels_pred = model.predict(X) # classify new
#     new_labels = Label_encoder.inverse_transform(labels_pred)
#     print(new_labels)
#     # for i in range(len(labels_pred)):
#     # print(X[i], new_labels[i])
#     # print(labels_pred.inverse_transform())


def logistic_regression_classifier():
    model = joblib.load("lr_polarity.pkl")  # load
    labels_pred = model.predict(X)  # classify new
    new_labels = Label_encoder.inverse_transform(labels_pred)
    print(df2.head())
    df2["ML_Polarity"] = new_labels
    print(df2.head())
    df2.to_csv("Tripadvisor_hotel_reviews-Rating-Polarity-PossibleIrony-MLPolarity(0-1).csv", index=False)
    # for i in range(len(labels_pred)):
    #     print(X2[i], new_labels[i])


read_tripadvisor_dataset()

#polarity experiments
print("EXPERIMENTS CROSS-VALIDATION FOLD=10---------------------------------")
print("RF---------------------------------")
random_forest_exp()
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
random_forest_model_training()
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
# print("LR---------------------------------")
# logistic_regression_classifier()

# distribution(df2, 'distrib_tripadvisor_irony.csv', 'Review', '')
