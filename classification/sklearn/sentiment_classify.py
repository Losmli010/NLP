# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np

def preprocessing():
    print("Load data...")
    pos_file = open("data/positive_sentiment.txt", "r", encoding="utf-8")
    neg_file = open("data/negative_sentiment.txt", "r", encoding="utf-8")
    data = [line.strip() for line in pos_file]
    data += [line.strip() for line in neg_file]
    pos_file.close()
    neg_file.close()
    length = int(len(data) / 2)
    labels = list(np.ones((length), dtype=np.int16))
    labels += list(np.zeros((length), dtype=np.int16))
    return data,labels

def split_data(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    data = data[shuffle_indices]
    labels = labels[shuffle_indices]
    
    # Split train/test set
    dev_sample_index = -1 * int(0.3 * float(len(labels)))
    train_data, test_data = data[:dev_sample_index], data[dev_sample_index:]
    y_train, y_test = labels[:dev_sample_index], labels[dev_sample_index:]
    
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return train_data, test_data, y_train, y_test

def train():
    data,labels = preprocessing()
    train_data, test_data, y_train, y_test = split_data(data, labels)
    
    vectorizer = CountVectorizer(max_df=0.5, min_df=1, stop_words=None)
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    X = vectorizer.transform(data)
#    print(vectorizer.get_feature_names())
    
    NBclf = MultinomialNB()
    NBclf.fit(X_train, y_train)
    print ("多项式贝叶斯分类器交叉验证得分: ", NBclf.score(X_test, y_test))
    print("多项式贝叶斯分类器准确率: ", accuracy_score(labels, NBclf.predict(X)))
    
    BNBclf = BernoulliNB()
    BNBclf.fit(X_train, y_train)
    print ("伯努利贝叶斯分类器交叉验证得分: ", BNBclf.score(X_test, y_test))
    print("伯努利贝叶斯分类器准确率: ", accuracy_score(labels, BNBclf.predict(X)))
    
    LRclf = LogisticRegression()
    LRclf.fit(X_train, y_train)
    print ("罗吉斯回归分类器交叉验证得分: ", LRclf.score(X_test, y_test))
    print("罗吉斯回归二分类器准确率: ", accuracy_score(labels, LRclf.predict(X)))
    
    SVMclf = svm.SVC()
    SVMclf.fit(X_train, y_train)
    print ("支持向量机分类器交叉验证得分: ", SVMclf.score(X_test, y_test))
    print("支持向量机二分类器准确率: ", accuracy_score(labels, SVMclf.predict(X)))

from sklearn.feature_extraction.text import TfidfVectorizer

def train_idf():
    data,labels = preprocessing()
    train_data, test_data, y_train, y_test = split_data(data, labels)
    
#    stop_words = [line.strip() for line in open("stop_words/stop_words.txt", "r", encoding="utf-8")]
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words=None, use_idf=True)
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    X = vectorizer.transform(data)
    
    NBclf = MultinomialNB()
    NBclf.fit(X_train, y_train)
    print ("多项式贝叶斯分类器交叉验证得分: ", NBclf.score(X_test, y_test))
    print("多项式贝叶斯分类器准确率: ", accuracy_score(labels, NBclf.predict(X)))
    
    BNBclf = BernoulliNB()
    BNBclf.fit(X_train, y_train)
    print ("伯努利贝叶斯分类器交叉验证得分: ", BNBclf.score(X_test, y_test))
    print("伯努利贝叶斯分类器准确率: ", accuracy_score(labels, BNBclf.predict(X)))
    
    LRclf = LogisticRegression()
    LRclf.fit(X_train, y_train)
    print ("罗吉斯回归分类器交叉验证得分: ", LRclf.score(X_test, y_test))
    print("罗吉斯回归二分类器准确率: ", accuracy_score(labels, LRclf.predict(X)))
    
    SVMclf = svm.SVC()
    SVMclf.fit(X_train, y_train)
    print ("支持向量机分类器交叉验证得分: ", SVMclf.score(X_test, y_test))
    print("支持向量机二分类器准确率: ", accuracy_score(labels, SVMclf.predict(X)))

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def train_chi2():
    data,labels = preprocessing()
    train_data, test_data, y_train, y_test = split_data(data, labels)
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=0.5, min_df=1, stop_words=None)
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    X = vectorizer.transform(data)
    
    ch2 = SelectKBest(chi2, k=50000)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    X = ch2.transform(X)
    
    NBclf = MultinomialNB()
    NBclf.fit(X_train, y_train)
    print ("多项式贝叶斯分类器交叉验证得分: ", NBclf.score(X_test, y_test))
    print("多项式贝叶斯分类器准确率: ", accuracy_score(labels, NBclf.predict(X)))
    
    BNBclf = BernoulliNB()
    BNBclf.fit(X_train, y_train)
    print ("伯努利贝叶斯分类器交叉验证得分: ", BNBclf.score(X_test, y_test))
    print("伯努利贝叶斯分类器准确率: ", accuracy_score(labels, BNBclf.predict(X)))
    
    LRclf = LogisticRegression()
    LRclf.fit(X_train, y_train)
    print ("罗吉斯回归分类器交叉验证得分: ", LRclf.score(X_test, y_test))
    print("罗吉斯回归二分类器准确率: ", accuracy_score(labels, LRclf.predict(X)))
    
    SVMclf = svm.SVC()
    SVMclf.fit(X_train, y_train)
    print ("支持向量机分类器交叉验证得分: ", SVMclf.score(X_test, y_test))
    print("支持向量机二分类器准确率: ", accuracy_score(labels, SVMclf.predict(X)))
