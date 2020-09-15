"""This learned bloom filter is mainly for testing the relationship between
% of memory each filter occupied vs. the general error rate
"""
import numpy as np
import Bloom_filters_test as BF
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import tqdm
import itertools
"""DATA PREPARATION"""
"""Check"""
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

#Run to show model error rate with different threshold
# err_array=[]
# for i in np.arange(0.1,1.0,0.01):
#     model_result=(probability_list>=i)
#     model_err=sum(np.not_equal(model_result,np.array(test_spam_df.iloc[:,-1])))/len(spam_df.iloc[:,-1])
#     err_array.append(model_err)
#
#
# plt.plot(np.arange(0, len(err_array)),err_array)
# plt.xlabel('Threshold Value')
# plt.ylabel('Model Error Rate')
# plt.title('Model Independent Error Rate vs Threshold Value')
# plt.show()


def learned_filter(m1,m2,t):
    #Space reserve for the two bloom filters
    """Initial Filter"""
    initial_bf=BF.Bloom_Filter(m1,len(all_pos_url))
    for i in range(len(all_pos_url)):
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
    backup_bf=BF.Bloom_Filter(m2,len(pos_backup_filter_data))

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

learned_filter(8000,1000,0.6)


initial_err_array=[]
ideal_initial_err_array=[]
backup_err_array=[]
ideal_backup_err_array=[]
total_err_array=[]
ideal_total_err_array=[]

for i in range(8000,18000,2000):
    total_err=[]
    for j in np.arange(0.1,1.0,0.02):
        temp_initial=[]
        temp_ideal_initial=[]
        temp_backup=[]
        temp_ideal_backup=[]
        temp_total=[]
        temp_ideal_total=[]
        for n in range(10):
            temp1,temp2,temp3,temp4,temp5,temp6=learned_filter(i,0.6)
            temp_initial.append(temp1)
            temp_ideal_initial.append(temp2)
            temp_backup.append(temp3)
            temp_ideal_backup.append(temp4)
            temp_total.append(temp1)
            temp_ideal_total.append(temp6)
        initial_err_array.append(np.median(np.array(temp_initial)))
        ideal_initial_err_array.append(np.median(np.array(temp_ideal_initial)))
        backup_err_array.append(np.median(np.array(temp_backup)))
        ideal_backup_err_array.append(np.median(np.array(temp_ideal_backup)))
        total_err.append(np.median(np.array(temp_total)))
        ideal_total_err_array.append(np.median(np.array(temp_ideal_total)))
    total_err_array.append(total_err)

initial_err_array=np.array(initial_err_array)
ideal_initial_err_array=np.array(ideal_initial_err_array)
backup_err_array=np.array(backup_err_array)
ideal_backup_err_array=np.array(ideal_backup_err_array)
total_err_array=np.array(total_err_array)
ideal_total_err_array=np.array(ideal_total_err_array)

#======================================================
"""For Multi subplots"""
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharey=True,figsize=(15,6))
fig.suptitle('How is Total Error Rate of the Learned Bloom Filter Change Regards Error Rate of the Different Sections',size=20)
ax1.plot(np.arange(0.1,1.0,0.01),initial_err_array,linestyle='solid', ms=20,label='Initial BF False Positive Rate')
ax1.plot(np.arange(0.1,1.0,0.01),ideal_initial_err_array, linestyle='dashdot', ms=20, label='Ideal Initial BF False Positive Rate')
ax2.plot(np.arange(0.1,1.0,0.01),backup_err_array, linestyle='solid', ms=20, label='Backup BF False Positive Rate')
ax2.plot(np.arange(0.1,1.0,0.01),ideal_backup_err_array, linestyle='dashdot', ms=20, label='Ideal Backup BF False Positive Rate')
ax3.plot(np.arange(0.1,1.0,0.01),total_err_array, linestyle='solid', ms=20, label='Total False Positive Rate')
ax3.plot(np.arange(0.1,1.0,0.01),ideal_total_err_array, linestyle='dashdot', ms=20, label='Ideal Total False Positive Rate')
fig.tight_layout(pad=3)
ax1.set_xlabel('% of Memory for Initial BF')
ax1.set_ylabel('Error Rate')
ax1.legend()
ax2.set_xlabel('% of Memory for Initial BF')
ax2.set_ylabel('Error Rate')
ax2.legend()
ax3.set_xlabel('% of Memory for Initial BF')
ax3.set_ylabel('Error Rate')
ax3.legend()
plt.savefig('result1.pdf')
plt.show()
1`#=================================================================
"""For multi lines plots"""
fig, ax = plt.subplots()

for i in range(len(total_err_array)):
    ax.plot(memory_array[i],total_err_array[i], linestyle='dotted', ms=15, label='Memory Size: '+str(np.arange(8000,18000,2000)[i]))

#ax.plot(np.arange(0,1.0,0.01),backup_err_array, linestyle='solid', ms=15, label='Back Up Filter Error Rate')
#ax.plot(np.arange(0,1.0,0.01),total_err_array, linestyle='dashdot', ms=15, label='Total Error Rate')
#ax.plot(np.arange(0,1.0,0.01),err_result_array, linestyle='dashdot', ms=15, label='Initial Filter Error Rate')
# ax.set_xlabel('Threshold Value')
# ax.set_ylabel('Model Error Rate')
ax.legend(loc=1,fontsize=8)
# ax2=ax.twinx()
# ax2.plot(np.arange(0.1,0.9,0.01),pos_count_array, linestyle='dashdot', ms=15, label='Predicted Positive Count')
# ax2.plot(np.arange(0.1,0.9,0.01),neg_count_array, linestyle='dashdot', ms=15, label='Predicted Negative Count')
# ax2.plot(np.arange(0.1,0.9,0.01),pos_truth_array, linestyle='solid', ms=15, label='Real Positive Count')
# ax2.plot(np.arange(0.1,0.9,0.01),neg_truth_array, linestyle='solid', ms=15, label='Real Negative Count')
# ax2.set_ylabel('Count')
# ax2.legend(loc=7,fontsize=8)
# ax.plot(np.arange(5000,20000,500),gradient, linestyle='dashdot', ms=15)
plt.xlabel('Backup Filter Memory')
plt.ylabel('Total Error Rate')
# plt.ylabel('Error Rate')
plt.title('Backup Filter Memory vs Total Error Rate with Memory Variations')
plt.savefig('result.pdf')
plt.show()
#=========================================================
"""For Grouped Dot Plots"""
# import scipy.optimize as optimize
# def curve_func(x,a,b,c):
#    return a*np.exp(-b*x)-c
# popt, pcov = optimize.curve_fit(curve_func, np.arange(0.99,1.0,0.0001), total_err_array,maxfev=100000)
# fitted_curve = curve_func(np.arange(0.99,1.0,0.0001), *popt)
# fig, ax = plt.subplots()
# #declare a counter to filter out different sample classes
# index=0
# #for loop to plot grouped plot
# for num in set(hash_result_array):
#     ax.plot(np.arange(0,1.0,0.01)[index:index+len(hash_result_array[hash_result_array==num])]
#     ,total_err_array[index:index+len(hash_result_array[hash_result_array==num])],
#             marker='o', linestyle='', ms=8,label='# of Hash '+str(num))
#     index=index+len(hash_result_array[hash_result_array==num])
# #ax.plot(np.arange(0.99,1.0,0.0001),fitted_curve,linestyle='-',ms=8,label='Fitted Curve')
# ax.legend()
plt.scatter(np.arange(0.1,1.0,0.01),total_err_array)
plt.xlabel('% of Memory for Initial BF')
plt.ylabel('Total Error Rate')
plt.title('% of Memory for Initial BF from 0.1 to 1.0 with Interval 0.01 with 12000 Memory')
plt.legend(loc=1,fontsize=6)
plt.savefig('rouhded_result.pdf')
plt.show()
