# coding: utf-8

import os
import sys
import logging

import re
from six import iteritems
from gensim import corpora, models


def get_dict(filename, dictname):
    curdir = sys.path[0]
    f = open(os.path.join(curdir,"data",filename), "r", encoding="utf-8")
    dictionary = corpora.Dictionary(line.strip().split() for line in f)
    f.close()
    
    stopwords_dir = os.path.join(curdir, "stop_words", "stop_words.txt")
    stop_words = [word.strip() for word in open(stopwords_dir, "r", encoding="utf-8")]
    stop_ids = [dictionary.token2id[stopword] for stopword in stop_words if stopword in dictionary.token2id]
    non_zh_ids = [dictionary.token2id[word] for word,_ in dictionary.token2id.items() if re.match(r"[^\u4E00-\u9FA5]+", word)]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids + non_zh_ids)
    dictionary.compactify()
    print(dictionary)
    dictionary.save(os.path.join(curdir, dictname))
    print("Save dictionary to %s for later use" % dictname)


def get_lda(dictname, tfidfname, ldaname, num_topics):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    curdir = sys.path[0]
    dictdir = os.path.join(curdir, "test", dictname)
    dictionary = corpora.Dictionary.load(dictdir)
    print(dictionary)
    tfidfdir = os.path.join(curdir, "test", tfidfname)
    corpus_tfidf = corpora.MmCorpus(tfidfdir)
    print(corpus_tfidf)
    lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
    ldadir = os.path.join(curdir, "test", ldaname)
    lda.save(ldadir)
    print("Save lda model to %s" % ldaname)


if __name__=="__main__":
    get_lda("zhwiki.dict", "zhwiki.tfidf", "zhwiki.lda.model", 100)
