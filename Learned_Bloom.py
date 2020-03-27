import numpy as np

"""DATA PREPARATION"""

"""Construct the Spam email array"""
spam_list=[]
spam_data=open('spambase.data','r')
for line in spam_data:
    spam_list.append(line.split(','))
spam_array=np.array(spam_list)
np.random.shuffle(spam_array)


"""Construct the url array"""
url_list=[]
url_data=open('Day117.svm','r')
for line in url_data:
    url_list.append(line.split(' '))
url_array=np.array(url_list)
index_array=np.arange(20000)
np.random.shuffle(index_array)
index_array=index_array[0:4601]
url_array=url_array[index_array]

np.concatenate(spam_array[spam_array])
