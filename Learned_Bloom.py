import numpy as np
import Bloom_filters as BF
from sklearn.naive_bayes import BernoulliNB

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
all_pos_url=np.array(malicious_url[0:1500])
"""Spam Email dataset """
all_pos_email=np.array(spam_email[0:1500])

"""All non-spam email dataset"""
all_neg_email=np.array(neg_email[0:2000])
"""All non-malicious URL"""
all_neg_url=np.array(neg_url[0:2000])

"""All Email dataset"""
all_email=np.concatenate((all_pos_email,all_neg_email),axis=0).astype(float)
"""All URL Dataset"""
all_url=np.concatenate((all_pos_url,all_neg_url),axis=0)

"""END OF DATA PREPARATION"""

"""CONSTRUCTING LEARNED BLOOM FILTER"""
"""INITIAL BF"""
initial_bf=BF.Bloom_Filter(0.20,1500)
for i in range(1500):
    initial_bf.insert(str(all_pos_url[i]))

"""ML CLASSIFIER"""
nb_classifier=BernoulliNB()
nb_classifier.fit(all_email[:,0:-1],all_email[:,-1])

"""BACK UP BF"""
backup_bf=BF.Bloom_Filter(0.05,1500)
for i in range(1500):
    backup_bf.insert(str(all_email[i]))

"""Test"""
round1_pos=[]
for i in range(len(all_url)):
    if initial_bf.search(all_url[i])==True:
        round1_pos.append(i)
classifier_email=all_email[round1_pos]
classifier_result=nb_classifier.predict(classifier_email[:,0:-1])
backupBF_result=[]
for i in range(len(classifier_email[classifier_result==0])):
    backupBF_result.append(backup_bf.search(classifier_email[classifier_result==0][i]))
print(' The input data set has 3500 instances in total, with 2000 benign/non-spam instances and 1500 spam/malicious data\n',
    'Initial Filter returns',len(round1_pos),'positives with',len(round1_pos)-1500,' false positives reported and discards', 3500-len(round1_pos),'negatives.\n',
    'The classifier takes in',len(round1_pos),'input email data instances and predicts that',int(sum(classifier_result)),'are spam email and',len(classifier_result)-int(sum(classifier_result)),'are non-spam email.\n',
    'The Backup Filter takes in',len(classifier_email[classifier_result==0]),'input email data instances. It predicts that',sum(backupBF_result),'instances are spam email and',len(classifier_email[classifier_result==0])-sum(backupBF_result),'are non-spam emails.\n',
    'Thus, as a conclusion:\n',
    'The Learned Filter predicted that there are',(3500-len(round1_pos))+(len(backupBF_result)-sum(backupBF_result)),'benign data instances and',int(sum(classifier_result))+sum(backupBF_result),'spam/malicioous data instances.')

"""RESULT"""
""" The input data set has 3500 instances in total, with 2000 benign/non-spam instances and 1500 spam/malicious data
 Initial Filter returns 1614 positives with 114  false positives reported and discards 1886 negatives.
 The classifier takes in 1614 input email data instances and predicts that 648 are spam email and 966 are non-spam email.
 The Backup Filter takes in 966 input email data instances. It predicts that 460 instances are spam email and 506 are non-spam emails.
 Thus, as a conclusion:
 The Learned Filter predicted that there are 2392 benign data instances and 1108 spam/malicioous data instances.
"""
