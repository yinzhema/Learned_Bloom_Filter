"""This learned bloom filter is mainly for testing the relationship between
% of memory each filter occupied vs. the general error rate
"""

import numpy as np
import Bloom_filters_test as BF
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
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

def learned_filter(pct):
    """CONSTRUCTING LEARNED BLOOM FILTER"""
    """INITIAL BF"""
    initial_bf=BF.Bloom_Filter(pct,30000,1500)
    for i in range(1500):
        initial_bf.insert(str(all_pos_url[i]))

    """ML CLASSIFIER"""
    """Data Cleansing"""
    training_email=all_email[(all_email[:,-2]<=4000) & (all_email[:,-3]<=1500) & (all_email[:,-4]<=200)]

    """Model"""
    nb_classifier=GaussianNB(var_smoothing=0)
    nb_classifier.fit(all_email[:,0:-1],all_email[:,-1])

    """BACK UP BF"""
    backup_bf=BF.Bloom_Filter(1-pct,30000,1500)
    for i in range(1500):
        backup_bf.insert(str(all_pos_url[i]))

    """Test"""
    round1_pos=[]
    for i in range(len(all_url)):
        if initial_bf.search(all_url[i])==True:
            round1_pos.append(i)
    classifier_email=all_email[round1_pos]
    classifier_result=nb_classifier.predict(classifier_email[:,0:-1])
    backup_bf_url=all_url[round1_pos]
    backupBF_result=[]
    for i in range(len(backup_bf_url[classifier_result!=1])):
        backupBF_result.append(backup_bf.search(backup_bf_url[classifier_result!=1][i]))
    error_rt=(3500-len(round1_pos)+(len(backupBF_result)-sum(backupBF_result)-2000))/3500

    return error_rt

result_array=[]
for i in np.arange(0.1,0.9,0.01):
    temp=[]
    for n in range(10):
        temp.append(learned_filter(i))
    result_array.append(np.average(np.array(temp)))

plt.scatter(np.arange(0.1,0.9,0.01),result_array)
plt.xlabel('% of Memory for Initial BF')
plt.ylabel('Error Rate')
plt.title('% of Memory for Initial BF from 0.1 to 0.9 with Interval of 0.01')
plt.show()
