"""Modified version of original Bloom Filter where Memory is given, and the % of Mermory this specific filter can
occupied is given; then the false positive rate is calculated
"""

"""Use murmur package for hash functions"""
import mmh3

import numpy as np

import math
"""Bloom Filter Class"""
class Bloom_Filter:
    """Inititalization function that takes in the false positive rate
        and number of elements to be inserted"""
    def __init__(self,mem,count,segments):
        self.array_size=mem

        self.length=count

        """Initiate a numpy array with zeros"""
        self.array=np.zeros(self.array_size,dtype=int)

        """calculate the optimal number of hash functions needed"""
        #Since g<=2k, we will set 2k as the max number of hash functions for on segment
        self.max_hash_num=2*int(math.floor((self.array_size/self.length)*np.log(2)))

        """Number of segments"""
        #self.segments=self.max_hash_num
        self.segments=segments

        """A dictionary to keep track of distribution of input data as in how many points in segment 1..."""
        self.distribution_dict={segment:0 for segment in range(self.segments)}

        """False positive rate; to be calculated after insertion"""
        self.fp_rate=0

    """Insertion method"""
    def insert(self,item,segment_id):
        """Number of has numbers needed for that segment"""
        hash_num=self.segments-segment_id

        """Increment by 1 in the distribution dictionary"""
        self.distribution_dict[segment_id]=self.distribution_dict[segment_id]+1

        """hash the item to be inserted to different hash functions with different seed numbers"""
        for i in range(hash_num):

            """The result mod the size of the array to get the index"""
            index=mmh3.hash(item,i)%(self.array_size)

            """set the specific index to 1"""
            self.array[index]=1
        return

    """Search method"""
    def search(self,item,segment_id):
        hash_num=self.segments-segment_id
        """Variable to indicate if the item is in the array"""
        found=0

        for i in range(hash_num):
            index=mmh3.hash(item,i)%(self.array_size)

            """If all the indices is already 1, then found=True"""
            if self.array[index]==1:
                found=found+1
            else:
                None

        return found==hash_num

    """Calculate the false positive rate"""
    def fp_calculator(self):

        fp_val=0

        """For loop to calculate each segment's false positive rate"""
        for i in range(self.segments):
            fp_val=fp_val+((self.distribution_dict[i]/self.array_size)*(1-math.exp((-(self.max_hash_num-i)*self.length)/self.array_size))**(self.max_hash_num-i))

        self.fp_rate=fp_val

        return fp_val
