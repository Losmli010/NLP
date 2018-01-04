# coding: utf-8

import codecs

def read_file(path,encoding="gbk"):
    file=codecs.open(path,"r",encoding)
    text=[]
    for line in file:
        line=line.strip()
        if line:
            text.append(line)
    file.close()
    return text

def write_file(path,text,encoding="utf-8"):
    file=codecs.open(path,"a",encoding)
    content=[line+"\n" for line in text]
    for line in content:
        file.write(line)
    file.close()

def gbk2utf8(src,dst):
    text = read_file(src,encoding="gbk")
    write_file(dst,text,encoding="utf-8")

