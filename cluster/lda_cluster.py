# coding: utf-8

import re
from six import iteritems
from gensim import corpora, models


def get_dict(filepath, dictpath):
    f = open(filepath, "r", encoding="utf-8")
    dictionary = corpora.Dictionary(line.strip().split() for line in f)
    f.close()
    
    stop_words = [word.strip() for word in open("stop_words/stop_words.txt", "r", encoding="utf-8")]
    stop_ids = [dictionary.token2id[stopword] for stopword in stop_words if stopword in dictionary.token2id]
    non_zh_ids = [dictionary.token2id[word] for word,_ in dictionary.token2id.items() if re.match(r"[^\u4E00-\u9FA5]+", word)]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids + non_zh_ids)
    dictionary.compactify()
    print(dictionary)
    dictionary.save(dictpath)
    print("Save dictionary to %s for later use" % dictpath)


def _iter_(filepath, dictpath):
    dictionary = corpora.Dictionary.load(dictpath)
    f = open(filepath, "r", encoding="utf-8")
    for line in f:
        line = line.strip().split()
        yield dictionary.doc2bow(line)


def get_corpus(filepath, dictpath, corpuspath):
    corpus_iter = _iter_(filepath, dictpath)
    corpus_tf = [vector for vector in corpus_iter]
    print("Corpus example:%s" % corpus_tf[:5])
    corpora.MmCorpus.serialize(corpuspath, corpus_tf)
    print("Save corpus to %s for later use" % corpuspath)


def get_tfidf(dictpath, corpuspath, tfidfpath):
    dictionary = corpora.Dictionary.load(dictpath)
    print(dictionary)
    corpus_tf = corpora.MmCorpus(corpuspath)
    print(corpus_tf)
    tfidf_model = models.TfidfModel(corpus_tf)
    corpus_tfidf = tfidf_model[corpus_tf]
    corpora.MmCorpus.serialize(tfidfpath, corpus_tfidf)
    print("Save tfidf to %s for later use" % tfidfpath)


def get_lsi(dictpath, tfidfpath, lsipath, num_topics):
    dictionary = corpora.Dictionary.load(dictpath)
    print(dictionary)
    corpus_tfidf = corpora.MmCorpus(tfidfpath)
    print(corpus_tfidf)
    lsi = models.lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    lsi.save(lsipath)
    print("Save lsi model to %s" % lsipath)
    

def get_lda(dictpath, tfidfpath, ldapath, num_topics):
    dictionary = corpora.Dictionary.load(dictpath)
    print(dictionary)
    corpus_tfidf = corpora.MmCorpus(tfidfpath)
    print(corpus_tfidf)
    lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
    lda.save(ldapath)
    print("Save lda model to %s" % ldapath)


def apply_labels(tfidfpath, ldapath, labelspath):
    corpus_tfidf = corpora.MmCorpus(tfidfpath)
    lda = models.LdaModel.load(ldapath)
    f = open(labelspath, "w", encoding="utf-8")
    for tfidf in corpus_tfidf:
        if tfidf:
            label,_ = sorted(lda[tfidf], key=lambda tup:tup[1], reverse=True)[0]
            f.write(str(label) + "\n")
        else:
            f.write("未分类" + "\n")
    f.close()
    print("Save label of documents to %s for classification" % labelspath)
