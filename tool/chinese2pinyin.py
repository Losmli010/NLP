# coding: utf-8

def chinese_to_pinyin(words,path):
    pinyin = ''
    chinese_unicode_pinyin= {}
    with open(path) as file:
        for i in file.readlines():
            chinese_unicode_pinyin[i.split()[0]] = i.split()[1]
    for word in words:
        word_unicode = str(word.encode('unicode_escape'))[-5:-1].upper()
        try:
            pinyin+= chinese_unicode_pinyin[word_unicode] + ' '
        except:
            pinyin += 'XXXX ' #非法字符我们用XXXX代替
    return pinyin

