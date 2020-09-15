"""This learned bloom filter is mainly for testing the relationship between
% of memory each filter occupied vs. the general error rate
"""
import numpy as np
import Bloom_filters_Modified as BF
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import tqdm
import itertools
"""DATA PREPARATION"""

"""Prepare the Spam Email Data"""
"""a list variable to store individual email data"""
spam_list=[]

"""read the data file and store them in spam_list"""
spam_data=open('spambase.data','r')
for line in spam_data:
    newline=[]
    for element in line.split(','):
        newline.append(float(element))
    spam_list.append(newline)

spam_df=pd.DataFrame(np.array(spam_list),columns=np.arange(len(spam_list[0])))



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

"""All non-malicious URL"""
all_neg_url=np.array(neg_url[0:2000])

"""All URL Dataset"""
all_url=np.concatenate((all_pos_url,all_neg_url),axis=0)
"""All Email Data Set"""
test_spam_df=pd.concat([spam_df.iloc[:1500,:],spam_df[spam_df[57]==0].iloc[:2000,:]])

"""END OF DATA PREPARATION"""
"""ML CLASSIFIER"""
"""Model"""
nb_classifier=MultinomialNB(alpha=1.0e-10)
nb_classifier.fit(spam_df.iloc[:,:-1],spam_df.iloc[:,-1])
probability_list=nb_classifier.predict_proba(test_spam_df.iloc[:,:-1])[:,-1]

def learned_filter(pct,t):
    #Space reserve for the two bloom filters
    """Initial Filter"""
    initial_bf=BF.Bloom_Filter(pct,12000,1500)
    for i in range(1500):
        initial_bf.insert(all_pos_url[i])

    #run the data through initial bloom filter
    round1_pos=[]
    for i in range(len(all_url)):
        if initial_bf.search(all_url[i])==True:
            round1_pos.append(i)
    #calculate the error rate of initial bf: False positive/All negatives
    initial_err=(len(round1_pos)-len(all_pos_url))/(len(all_url)-len(all_pos_url))

    """Naive Bayes Classifier"""
    # Oracle Model
    # np.array(round1_pos[:1500]).resample(frac=0.1)
    # c_result=np.array(classifier_email[:,-1])
    # counter=0
    # counter1=0
    # for i in range (len(c_result)):
    #     if c_result[i]==1.0 and counter<round(false_neg*len(c_result)):
    #         c_result[i]=0.0
    #         counter=counter+1
    #     elif c_result[i]==0.0 and counter1<round(false_pos*len(c_result)):
    #         c_result[i]=1.0
    #         counter1=counter1+1
    #using actual model prediction
    positive_array=(probability_list[round1_pos]>=t) #the data instance that are positive from initial filter and prob of 1 is bigger than T
    backup_filter_data=all_url[round1_pos][positive_array!=True]
    pos_backup_filter_data=[]
    for i in range(len(backup_filter_data)):
        if backup_filter_data[i][0]=='+1':
            pos_backup_filter_data.append(backup_filter_data[i])

    model_miss=sum(positive_array)-sum(np.array(test_spam_df[57])[round1_pos][positive_array])
    model_false_pos=model_miss/(len(positive_array)-sum(probability_list[round1_pos]))
    neg_array=np.invert(positive_array)
    neg_miss=sum(neg_array)-(len(np.array(test_spam_df[57])[round1_pos][neg_array])-sum(np.array(test_spam_df[57])[round1_pos][neg_array]))
    model_false_neg=neg_miss/sum(probability_list[round1_pos])

    """Backup Filter"""
    backup_bf=BF.Bloom_Filter(1-pct,12000,len(pos_backup_filter_data))

    for i in range(len(pos_backup_filter_data)):
        backup_bf.insert(pos_backup_filter_data[i])

    backup_bf_result=[]
    backup_bf_truth=[]
    for i in range(len(backup_filter_data)):
        backup_bf_result.append(backup_bf.search(backup_filter_data[i]))
        backup_bf_truth.append(int(backup_filter_data[i][0]))

    backup_err=(sum(backup_bf_result)-len(pos_backup_filter_data))/(len(backup_filter_data)-len(pos_backup_filter_data))
    """Total Error Rate"""
    #initial_miss=len(round1_pos)-len(all_pos_url)
    backup_miss=sum(backup_bf_result)-len(pos_backup_filter_data)
    total_err=(model_miss+backup_miss)/(2000)
    ideal_total_err=(0.6185**(initial_bf.array_size/initial_bf.length))*(model_false_pos+(1-model_false_pos)*(0.6185**((backup_bf.array_size/backup_bf.length))))

    return initial_err,initial_bf.false_pos,backup_err,backup_bf.false_pos, total_err,ideal_total_err
