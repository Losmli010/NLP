# coding: utf-8

import codecs

def read_file(path,encoding="gb18030"):
    file=codecs.open(path,"r",encoding)
    text=[line.strip() for line in file if line]
    file.close()
    return text

def write_file(path,text,encoding="utf-8"):
    file=codecs.open(path,"a",encoding)
    for line in text:
        line=line+"\n"
        file.write(line)
    file.close()

def gbk2utf8(src,dst):
    text = read_file(src,encoding="gb18030")
    write_file(dst,text,encoding="utf-8")
