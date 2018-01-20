# coding: utf-8

import random

class idiom_game(object):
    def __init__(self):
        self.idiom_pinyin = {}
        self.pinyin_idiom = {}
        self.inputs = ""
        self.outputs = ""
    
    def play(self):
        self.build_dict()
        self.outputs = random.choice(list(self.idiom_pinyin.keys()))
        print("我先来，%s，接招"% self.outputs)       
        self.inputs = input()
        
        times = 0
        wrong = 0
        while True:
            if self.exist():
                if self.rule():
                    try: 
                        self.search()
                        print("%s，感觉自己棒棒哒" % self.outputs)
                        self.update_dict()
                        self.inputs = input()
                    except:
                        print("厉害了，是在下输了")
                        break
                else:
                    wrong += 1
                if wrong == 3:
                    print("累计3次不遵守游戏规则，罚你抄写<<社会主义核心价值观>>10086遍")
                    break        
            else:
                times += 1   
            if times == 3:
                print("出错3次。你妈妈喊你回家背字典呐")
                break      
    
    def exist(self):
        if self.inputs in self.idiom_pinyin:
            return True
        else:
            print("这是成语么，你特么逗我")
            self.inputs = input()
            return False
        
    def rule(self):
        if self.idiom_pinyin[self.inputs].split()[0] != self.idiom_pinyin[self.outputs].split()[-1]:
            print("成语的第一个字必须和我的成语最后一个字是同音字哦")
            self.inputs = input()
            return False
        return True
                
    def build_dict(self):
        f = open("idioms.txt", "r", encoding="utf-8")
        for line in f:
            tokens = line.strip().split("||")
            #idioms dictionary, dict(idiom = pinyin)
            self.idiom_pinyin[tokens[0]] = tokens[1]
            
            #first word pinyin dictionary, dict(first_pinyin = [idiom1, idioms2])
            first_pinyin = tokens[1].split()[0]
            self.pinyin_idiom.setdefault(first_pinyin, [])
            self.pinyin_idiom[first_pinyin].append(tokens[0])
        f.close()
    
    def search(self):
        pinyin = self.idiom_pinyin[self.inputs]
        last_pinyin = pinyin.split()[-1]
        idioms = self.pinyin_idiom[last_pinyin]
        self.outputs = random.choice(idioms)
   
    def remove_idiom(self, idiom):
        pinyin = self.idiom_pinyin[idiom]
        first_pinyin = pinyin.split()[0]
        idiom_list = self.pinyin_idiom[first_pinyin]
        index = idiom_list.index(idiom)
        del self.pinyin_idiom[first_pinyin][index]
    
    def update_dict(self):
        self.remove_idiom(self.inputs)
        self.remove_idiom(self.outputs)
                    
if __name__=="__main__":
    game = idiom_game()
    game.play()