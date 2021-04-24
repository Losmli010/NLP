import os

from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
import jieba


LTP_DATA_DIR = "D:\\Anaconda\\Lib\\site-packages\\ltp\\3.4"
text = "元芳你怎么看？我就趴窗口上看呗！"


def split(text):
    return SentenceSplitter.split(text)


def cut(sentences):
    return jieba.cut(sentences)


def segment(sentences):
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")
    lexicon_path = os.path.join(LTP_DATA_DIR, "lexicon.txt")

    segmentor = Segmentor()
    # segmentor.load(cws_model_path)
    segmentor.load_with_lexicon(cws_model_path, lexicon_path)
    words = segmentor.segment(sentences)
    segmentor.release()
    return words


def pos(words):
    pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")

    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(words)
    postagger.release()
    return postags


def ner(words, postags):
    ner_model_path = os.path.join(LTP_DATA_DIR, "ner.model")

    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    netags = recognizer.recognize(words, postags)
    recognizer.release()
    return netags


def match(tokens, tag, tags, word, namedentity):
    if tokens[0] == "B":
        tags += [tag]
        namedentity += [word]
    elif tokens[0] == "I":
        tags += [tag]
        namedentity += [word]
    elif tokens[0] == "E":
        tags += [tag, " "]
        namedentity += [word, " "]
    else:
        tags += [tag, " "]
        namedentity += [word, " "]
    return tags, namedentity


def recognize(words, netags):
    namedentity = []
    tags = []
    for word, tag in zip(words, netags):
        if tag != "O":
            tokens = tag.split("-")
            if tokens[1] == "Nh":
                tags, namedentity = match(tokens, tag, tags, word, namedentity)
            elif tokens[1] == "Ni":
                tags, namedentity = match(tokens, tag, tags, word, namedentity)
            else:
                tags, namedentity = match(tokens, tag, tags, word, namedentity)
                
    namedentity = "".join(namedentity).split()
    tags = [tag.split("-")[-1] for tag in "".join(tags).split()]
    return dict(zip(namedentity, tags))
