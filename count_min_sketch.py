"""Use murmur package for hash functions"""
import mmh3

import numpy as np

import Helper

"""Count Min Sketch Class"""
class countMin_Sketch:
    """Inititalization function that takes in the false positive rate
        and number of elements to be inserted"""
    def __init__(self,error_rate,count):
        self.er_rate=error_rate

        self.length=count

        """Calculate the number of buckets"""
        self.bucket_num=int(1/self.er_rate)

        """calculate the optimal number of hash functions needed"""
        self.hash_num=int(np.log(1/self.er_rate))

        """Initiate a 2D numpy array with zeros"""
        self.array=np.zeros((self.hash_num, self.bucket_num),dtype=int)

    """Increment method"""
    def increment(self,item):

        """hash the item to be inserted to different hash functions with different seed numbers"""
        for i in range(self.hash_num):

            """The result mod the size of the array to get the index"""
            index=mmh3.hash(item,i)%(self.bucket_num)

            """increment the specific index by 1"""
            self.array[i][index]+=1
        return

    """Count method, return the min count# for a specific item"""
    def count(self,item):

        """a list to store all count # across different hash function"""
        countMin=[]

        """locate all count # then store them in countMin"""
        for i in range(self.hash_num):
            index=mmh3.hash(item,i)%(self.bucket_num)
            countMin.append(self.array[i][index])

        """return the min of countMin"""
        return min(countMin)

"""TESTING"""
helper=Helper.Helpers()
helper.read_txt("sample.txt")
countMin_test=countMin_Sketch(0.05,helper.word_count)

for line in helper.words:
    for word in line.split():
        countMin_test.increment(word)
