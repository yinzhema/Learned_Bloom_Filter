import mmh3
import numpy as np

class Bloom_Filter:
    def __init__(self,fp_rate,count):
        self.false_pos=fp_rate
        self.length=count
        self.array_size=int((-(self.length)*np.log(self.false_pos))/(np.log(2)**2))
        self.array=np.zeros(self.array_size,dtype=int)
        self.hash_num=int((self.array_size/self.length)*np.log(2))

    def insert(self,item):
        for i in range(self.hash_num):
            index=mmh3.hash(item,i)%(self.array_size)
            self.array[index]=1
        return

    def search(self,item):
        found=False
        for i in range(self.hash_num):
            index=mmh3.hash(item,i)%(self.array_size)
            if self.array[index]==1:
                found=True
            else:
                found=False
        return found

class BF_Helper:
    def __init__(self):
        self.title=''
        self.word_count=0
        self.words=[]

    def read_txt(self, title):
        wordCount=0
        file=open(title,"r")
        for line in file:
            wordCount+=len(line.split())
            self.words.append(line)
        self.word_count=wordCount
        self.title=title


bf_helper=BF_Helper()
bf_helper.read_txt('sample.txt')
bf_test=Bloom_Filter(0.05,bf_helper.word_count)

for line in bf_helper.words:
    print(line)
    for word in line.split():
        print(word)
        bf_test.insert(word)
