"""Modified version of original Bloom Filter where Memory is given, and the % of Mermory this specific filter can
occupied is given; then the false positive rate is calculated
"""

"""Use murmur package for hash functions"""
import mmh3

import numpy as np

import Helper

import math
"""Bloom Filter Class"""
class Bloom_Filter:
    """Inititalization function that takes in the false positive rate
        and number of elements to be inserted"""
    def __init__(self,memory_pct,memory,count):
        self.array_size=int(memory*memory_pct)

        self.length=count

        """Initiate a numpy array with zeros"""
        self.array=np.zeros(self.array_size,dtype=int)

        """calculate the optimal number of hash functions needed"""
        self.hash_num=int(round((self.array_size/self.length)*np.log(2)))

        """Calculate false positive rate"""
        self.false_pos=(1-math.exp((-self.hash_num*self.length)/self.array_size))**self.hash_num

    """Insertion method"""
    def insert(self,item):

        """hash the item to be inserted to different hash functions with different seed numbers"""
        for i in range(self.hash_num):

            """The result mod the size of the array to get the index"""
            index=mmh3.hash(item,i)%(self.array_size)

            """set the specific index to 1"""
            self.array[index]=1
        return

    """Search method"""
    def search(self,item):

        """Variable to indicate if the item is in the array"""
        found=0

        for i in range(self.hash_num):
            index=mmh3.hash(item,i)%(self.array_size)

            """If all the indices is already 1, then found=True"""
            if self.array[index]==1:
                found=found+1
            else:
                None

        return found==self.hash_num


bf_test=Bloom_Filter(1,12000,1500)
bf_test.false_pos
bf_test.array_size
bf_test.hash_num


