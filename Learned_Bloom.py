import numpy as np

"""DATA PREPARATION"""

"""Prepare the Spam Email Data"""
"""a list variable to store individual email data"""
spam_list=[]

"""read the data file and store them in spam_list"""
spam_data=open('spambase.data','r')
for line in spam_data:
    spam_list.append(line.split(','))

"""Seperate the spam_list dataset to spam email and non-spam email(neg_email) """
spam_email=[]
neg_email=[]

for i in range(len(spam_list)):
    if spam_list[i][-1]=='1\n':
        spam_email.append(spam_list[i])
    else:
        neg_email.append(spam_list[i])

"""Prepare the URL Data"""
"""a list variable to store individual url data"""
url_list=[]

"""read the data file and store them in url_data"""
url_data=open('Day117.svm','r')
for line in url_data:
    url_list.append(np.array(line.split(' ')))

"""transform the url_list to url_array"""
url_array=np.array(url_list)

"""create an array of index, and shuffle it randomly,
    then take the first 4601 data instances which is the same length as email data,
    and store them in url_array"""
index_array=np.arange(20000)
np.random.shuffle(index_array)
index_array=index_array[0:4601]
url_array=url_array[index_array]

"""seperate the url data into malicious set and non-malicious set"""
malicious_url=[]
neg_url=[]
for i in range(len(url_array)):
    if url_array[i][0]=='+1':
        malicious_url.append(url_array[i])
    else:
        neg_url.append(url_array[i])

"""Build up different datasets for filter"""
"""Malicious URL dataset"""
all_pos_url=np.array(malicious_url[0:1000])
"""Spam Email dataset """
all_pos_email=np.array(spam_email[0:1000])

"""False Positive Email dataset: half spam, half non-spam"""
false_pos_email=np.concatenate((np.array(spam_email[0:250]),np.array(neg_email[0:250])),axis=0)
"""False Positive URL dataset: All Malicious URL"""
false_pos_url=np.array(malicious_url[1000:1500])

"""All non-spam email dataset"""
all_neg_email=np.array(neg_email[250:2250])
"""All non-malicious URL"""
all_neg_url=np.array(neg_url[0:2000])

"""END OF DATA PREPARATION"""
