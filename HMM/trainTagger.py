# coding: utf-8

from utils import *
import pickle

def tags_set():
    tags=['Ag',          #形语素
          'Bg',          #区别语素
          'Dg',          #副语素
          'Mg',          #数语素
          'Ng',          #名语素
          'Rg',          #代语素
          'Tg',          #时语素
          'Vg',          #动语素
          'Yg',          #语气语素
          'a',           #形容词
          'ad',          #副形词
          'an',          #名形词
          'b',           #区别词
          'c',           #连词
          'd',           #副词
          'e',           #叹词
          'f',           #方位词
          'h',           #前接成分
          'i',           #成语
          'j',           #简称略语
          'k',           #后接成分
          'l',           #习用语
          'm',           #数词
          'n',           #名词
          'nr',          #人名
          'ns',          #地名
          'nt',          #机构团体
          'nx',          #外文字符
          'nz',          #其他专名
          'o',           #拟声词
          'p',           #介词
          'q',           #量词
          'r',           #代词
          's',           #处所词
          't',           #时间词
          'u',           #助词
          'v',           #动词
          'vd',          #副动词
          'vn',          #动名词
          'w',           #标点符号
#          'x',           #非语素字
          'y',           #语气词
          'z']           #状态词
    return tags

def process_corpus():
    """
    人民日报语料库预处理
    rtype:list(tuple(word,tag))
    >>>[('迈向', 'v'),('充满', 'v'),('希望', 'n'),('的', 'u'),('新', 'a'),('世纪', 'n'),
        ('——', 'w'),('一九九八年', 't'),('新年', 't'),('讲话', 'n'),('（', 'w'),
        ('附', 'v'),('图片', 'n'),('１', 'm'),('张', 'q'),('）', 'w'),
       ('中共中央', 'nt'),('总书记', 'n'),('、', 'w'),('国家', 'n'),('主席', 'n'),
        ('江', 'nr'),('泽民', 'nr')]
    """
    f=open("data/ThePeople'sDaily199801.txt",'r',encoding='utf-8')
    
    tagged_sents=[]
    for line in f:
        raw_word_tag=[tuple(word.split('/')) for word in line.split()[1:]]
        tagged_words =[(word.strip(),tag.strip()) for word,tag in raw_word_tag]
        tagged_sents.append(tagged_words)
    f.close()
    return tagged_sents

def train():
    tagged_sents=process_corpus()
    tagged_words=[(w,t) for sent in tagged_sents for w,t in sent]
    words_set=sorted(list(set([w for w,t in tagged_words])))
    words_set.append("UNK")
    labels_set=tags_set()
    emission_prob=cal_condition_prob(labels_set,words_set,tagged_words)
    
    labels=[t for w,t in tagged_words]
    symbol_label=Condition(labels)
    trans_prob=cal_condition_prob(labels_set,labels_set,symbol_label,discount=0.1)
    
    init_label=[t for w,t in [sent[0] for sent in tagged_sents if sent]]
    init_prob=ProbDist(AbsoluteDiscounting(FreqDist(labels_set,init_label),discount=0.01))
    save_parameters("data/HMMTagger.parameters.npz",init_prob,trans_prob,emission_prob)
    
    word_index=symbols2index(words_set)
    label_index=index2labels(labels_set)
    with open("data/Taggerindex.pkl","wb") as f:
        pickle.dump(word_index,f)
        pickle.dump(label_index,f)
    f.close()
