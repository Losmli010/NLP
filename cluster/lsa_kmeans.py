# coding: utf-8

import logging
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans


def doc_iter(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            yield line


def lsa_kmeans(raw_doc, n_components, n_clusters):
    logger = logging.getLogger("lsa_kmeans")
    logger.info("Loading stop words...")
    stop_words = [word.strip() for word in open("stop_words/stop_words.txt", "r", encoding="utf-8")]
    
    logger.info("Compute tf-idf")
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1),
                                 max_df=1.0, min_df=1, max_features=None,
                                 vocabulary=None, norm="l2", use_idf=True,
                                 smooth_idf=True, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(raw_doc)
    
    logger.info("SVD for lsa")
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized",
                       n_iter=5, random_state=None, tol=0.0)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tfidf)
    
    logger.info("clustering documents")
    clf = KMeans(algorithm="auto", copy_x=True, init="k-means++", max_iter=300,
                 n_clusters=n_clusters, n_init=10, n_jobs=-2, precompute_distances="auto",
                 random_state=None, tol=0.0001, verbose=0)
    clf.fit(X)
    return clf


def save_model(clf, path):
    print("Save KMeans model to %s" % path)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    f.close()
    

def load_model(path):
    print("Load KMeans model to %s" % path)
    with open(path, "rb") as f:
        clf = pickle.load(f)
    f.close()
    return clf


def categories():
    return ["life_art_culture",                 # 生活、艺术与文化
            "chinese_culture",                  # 中华文化
            "social",                           # 社会
            "religion_belief",                  # 宗教及信仰
            "world",                            # 世界各地
            "social_sciences",                  # 人文与社会科学
            "natural_science",                  # 自然与自然科学
            "engineering_technology",           # 工程、技术与应用科学
            "general_works",                    # 常用列表
            "topic"                             # 主题首页
           ]
