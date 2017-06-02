# coding=utf-8
from corpus import Corpus

corpus = Corpus("train_utf16.tag", 0.7)
corpus.train()
print(corpus.check())
print(corpus.tag(["總辦事處", "秘書組", "主任", "戴政", "先生", "請辭", "獲准", "，"]))
