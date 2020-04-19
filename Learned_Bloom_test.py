"""This learned bloom filter is mainly for testing the relationship between
% of memory each filter occupied vs. the general error rate
"""
import numpy as np
import Bloom_filters_Modified as BF
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

"""ML CLASSIFIER"""
"""Data Cleansing"""
#training_email=all_email[(all_email[:,-2]<=2000) & (all_email[:,-3]<=500) & (all_email[:,-4]<=25)]
"""Model"""
nb_classifier=GaussianNB(var_smoothing=0)
nb_classifier.fit(all_email[:,0:-1],all_email[:,-1])

def learned_filter(pct):
    #Space reserve for the two bloom filters
    """Initial Filter"""
    initial_bf=BF.Bloom_Filter(pct,10000,1500)
    for i in range(1500):
        initial_bf.insert(all_url[i])
    """Backup Filter"""
    backup_bf=BF.Bloom_Filter(1-pct,10000,1500)
    for i in range(1500):
        backup_bf.insert(all_url[i])
    #run the data through initial bloom filter
    round1_pos=[]
    for i in range(len(all_url)):
        if initial_bf.search(all_url[i])==True:
            round1_pos.append(i)
    #calculate the error rate of initial bf
    initial_err=abs((len(round1_pos)-len(all_pos_url)))/len(all_url)
    #Executet the classifier model
    classifier_email=all_email[round1_pos]
    # c_result=np.array(classifier_email[:,-1])
    # error_array=np.random.choice(np.arange(len(c_result)),size=int(round((false_pos+false_neg)*len(c_result))),replace=False)
    # for i in error_array:
    #     if c_result[i]==1.0:
    #         c_result[i]=0.0
    #     elif c_result[i]==0.0:
    #         c_result[i]=1.0
    #print(c_result!=1.0)
    c_result=nb_classifier.predict(classifier_email[:,0:-1])
    #calculate the model error rate
    model_err=sum(c_result!=classifier_email[:,-1])/len(classifier_email)
    #run the data through back up filter
    backup_bf_url=all_url[round1_pos][c_result!=1.0]
    #print(backup_bf_url)
    backupBF_result=[]
    backupBF_truth=[]
    for i in range(len(backup_bf_url)):
        backupBF_result.append(backup_bf.search(backup_bf_url[i]))
        backupBF_truth.append(int(backup_bf_url[i][0]))
    #calculate the backup filter error rate
    backup_err=sum(np.array(backupBF_result)!=(np.array(backupBF_truth)==True))/len(backupBF_truth)
    #calculate the whole data strucutre error rate
    error_rt=abs((3500-len(round1_pos))+(len(backupBF_result)-sum(backupBF_result))-2000)/3500

    return initial_err,model_err,backup_err,error_rt,initial_bf.hash_num

learned_filter(0.6)



backup_err_array=[]
err_result_array=[]
hash_result_array=[]
#false_pos_array=[]
#false_neg_array=[]
model_err_array=[]
total_err_array=[]
for i in np.arange(0.1,0.9,0.01):
    temp_err=[]
    temp_hash=[]
    temp_model=[]
    temp_backup=[]
    temp_total=[]
    for n in range(10):
        temp1,temp2,temp3,temp4,temp5 =learned_filter(i)
        temp_err.append(temp1)
        temp_hash.append(temp5)
        temp_model.append(temp2)
        temp_backup.append(temp3)
        temp_total.append(temp4)
    err_result_array.append(np.median(np.array(temp_err)))
    hash_result_array.append(np.median(np.array(temp_hash)))
    model_err_array.append(np.median(np.array(temp_model)))
    backup_err_array.append(np.median(np.array(temp_backup)))
    total_err_array.append(np.median(np.array(temp_total)))
    # false_neg_array.append(i*0.5)
    # false_pos_array.append(i*0.5)



err_result_array=np.array(err_result_array)
hash_result_array=np.array(hash_result_array)
model_err_array=np.array(model_err_array)
backup_err_array=np.array(backup_err_array)
total_err_array=np.array(total_err_array)
# false_pos_array=np.array(false_pos_array)
# false_neg_array=np.array(false_neg_array)

#======================================================
"""For Multi subplots"""
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharey=True,figsize=(15,6))
# fig.suptitle('How is Total Error Rate of the Learned Bloom Filter Regards Error Rate of the Different Sections',size=20)
# ax1.plot(np.arange(0.1,0.90,0.01),err_result_array,linestyle='solid', ms=20,label='Initial BF Error Rate')
# ax1.plot(np.arange(0.1,0.90,0.01),total_err_array, linestyle='dashdot', ms=20, label='Total Error Rate')
# ax2.plot(np.arange(0.1,0.90,0.01),model_err_array, linestyle='solid', ms=20, label='Model Error Rate')
# ax2.plot(np.arange(0.1,0.90,0.01),total_err_array, linestyle='dashdot', ms=20, label='Total Error Rate')
# ax3.plot(np.arange(0.1,0.90,0.01),backup_err_array, linestyle='solid', ms=20, label='Backup BF Error Rate')
# ax3.plot(np.arange(0.1,0.90,0.01),total_err_array, linestyle='dashdot', ms=20, label='Total Error Rate')
# fig.tight_layout(pad=3)
# ax1.set_xlabel('% of Memory for Initial BF')
# ax1.set_ylabel('Error Rate')
# ax1.legend()
# ax2.set_xlabel('% of Memory for Initial BF')
# ax2.set_ylabel('Error Rate')
# ax2.legend()
# ax3.set_xlabel('% of Memory for Initial BF')
# ax3.set_ylabel('Error Rate')
# ax3.legend()
# plt.savefig('result1.pdf')
# plt.show()
#=================================================================
"""For multi lines plots"""
# fig, ax = plt.subplots()
# #declare a counter to filter out different sample classes
# index=0
# #for loop to plot grouped plot
# ax.plot(np.arange(0.0,0.20,0.01),false_neg_array,linestyle='solid', ms=15,label='False Negative Rate')
# ax.plot(np.arange(0.0,0.20,0.01),model_err_array, linestyle='dotted', ms=15, label='Model Error Rate')
# ax.plot(np.arange(0.0,0.20,0.01),false_pos_array, linestyle='dashed', ms=15, label='False Positive Rate')
# ax.plot(np.arange(0.0,0.20,0.01),total_err_array, linestyle='dashdot', ms=15, label='Total Error Rate')
# ax.legend()
# plt.xlabel('% of Memory for Initial BF')
# plt.ylabel('Error Rate')
# plt.title('% of Memory for Initial BF from 0.1 to 0.9 with Interval 0.01 with # of Hash Functions--30000 Memory')
# plt.legend(loc=7,fontsize=8)
# plt.savefig('result.pdf')
# plt.show()
#=========================================================
"""For Grouped Dot Plots"""
# fig, ax = plt.subplots()
# #declare a counter to filter out different sample classes
# index=0
# #for loop to plot grouped plot
# for num in set(hash_result_array):
#     ax.plot(np.arange(0.1,0.90,0.01)[index:index+len(hash_result_array[hash_result_array==num])]
#     ,err_result_array[index:index+len(hash_result_array[hash_result_array==num])],
#             marker='o', linestyle='', ms=8,label='# of Hash '+str(num))
#     index=index+len(hash_result_array[hash_result_array==num])
#
# ax.legend()
# plt.xlabel('% of Memory for Initial BF')
# plt.ylabel('Error Rate')
# plt.title('% of Memory for Initial BF from 0.1 to 0.9 with Interval 0.01 with # of Hash Functions--10000 Memory')
# plt.legend(loc=1,fontsize=6)
# plt.savefig('rouhded_result.pdf')
# plt.show()
