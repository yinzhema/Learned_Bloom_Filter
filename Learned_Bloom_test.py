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


"""Seperate the spam_list dataset to spam email and non-spam email(neg_email) """
spam_email=[]
neg_email=[]

for i in range(len(spam_list)):
    if spam_list[i][-1]==1.0:
        spam_email.append(spam_list[i])
    else:
        neg_email.append(spam_list[i])

"""Prepare the URL Data"""
"""a list variable to store individual url data"""
url_list=[]

"""read the data file and change them to dataframe"""
url_data=open('Day117.svm','r')

for line in url_data:
    newline=[]
    for element in line.split(' '):
        newline.append([float(ele) for ele in element.split(':')])
    newline[0]=[0.0]+newline[0]
    newline=np.array(newline)
    url_list.append(pd.DataFrame(dict(zip(newline.T[0],newline.T[1])),index=[0]))


def df_concat(df_list):
    return pd.concat(df_list)

if __name__ == "__main__" :
    p=multiprocessing.Pool(processes=6)
    final_df_list=p.map(df_concat,[url_list[0:3333],url_list[3333:6666],url_list[6666:9999],url_list[9999:13332],url_list[13332:16665],url_list[16665:]])
    p.close()

final=pd.concat(final_df_list)
final=final.reset_index().sample(frac=1)
final.to_csv('url_data.csv')

"""TO BE UPGRADED"""
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
nb_classifier=MultinomialNB(alpha=1.0e-10)
nb_classifier.fit(all_email[:,0:-1],all_email[:,-1])
probability_list=nb_classifier.predict_proba(all_email[:,0:-1])
err_array=[]
for i in np.arange(0.1,1.0,0.01):
    c_result=(probability_list[:,1]>=i)
    model_err=sum(c_result!=all_email[:,-1])/len(all_email)
    err_array.append(model_err)
# plt.scatter(np.arange(0, len(probability_list)),probability_list[:,1])
# plt.xlabel('Threshold Value')
# plt.ylabel('Model Error Rate')
# plt.title('Model Independent Error Rate vs Threshold Value')
# plt.show()
sum(all_email[:,-1]==True)

def learned_filter(pct,t):
    #Space reserve for the two bloom filters
    """Initial Filter"""
    initial_bf=BF.Bloom_Filter(pct,6000,1500)
    for i in range(1500):
        initial_bf.insert(all_url[i])
    """Backup Filter"""
    backup_bf=BF.Bloom_Filter(1-pct,6000,1500)
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
    classifier_email=all_email[round1_pos].copy()
    #Oracle Model
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
    #Use Model prediction instead of T
    #c_result=nb_classifier.predict(classifier_email[:,0:-1])
    #Use T instead of model prediction
    probability_list=nb_classifier.predict_proba(classifier_email[:,0:-1])
    c_result=(probability_list[:,1]>=t)
    pos_count=sum(c_result)
    neg_count=len(c_result)-pos_count
    pos_truth=sum(classifier_email[:,-1]==True)
    neg_truth=len(classifier_email)-pos_truth
    # print(len(c_result))
    #calculate the model error rate
    model_err=sum(c_result!=classifier_email[:,-1])/len(classifier_email)
    #run the data through back up filter
    backup_bf_url=all_url[round1_pos][c_result!=True]
    #print(backup_bf_url)
    backupBF_result=[]
    backupBF_truth=[]
    for i in range(len(backup_bf_url)):
        backupBF_result.append(backup_bf.search(backup_bf_url[i]))
        backupBF_truth.append(int(backup_bf_url[i][0]))
    #calculate the backup filter error rate
    backup_err=sum(np.array(backupBF_result)!=(np.array(backupBF_truth)==True))/len(backupBF_truth)
    #calculate the whole data strucutre error rate
    error_rt=(abs((3500-len(round1_pos))+(len(backupBF_result)-sum(backupBF_result))-2000)+abs(len(c_result[c_result==True])+sum(backupBF_result)-1500))/3500
    #return initial_err,model_err,backup_err,error_rt,initial_bf.hash_num
    return model_err, pos_count,neg_count,pos_truth, neg_truth

learned_filter(0.96,0.8)

#
# backup_err_array=[]
# err_result_array=[]
# hash_result_array=[]
model_err_array=[]
pos_count_array=[]
neg_count_array=[]
pos_truth_array=[]
neg_truth_array=[]
#total_err_array=[]

for i in np.arange(0.1,0.9,0.01):
    # temp_err=[]
    # temp_hash=[]
    temp_model=[]
    # temp_backup=[]
    # temp_total=[]
    temp_pos_count=[]
    temp_neg_count=[]
    temp_pos_truth=[]
    temp_neg_truth=[]
    for n in range(10):
        temp1,temp2,temp3,temp4,temp5=learned_filter(0.75,i)
        temp_model.append(temp1)
        temp_pos_count.append(temp2)
        temp_neg_count.append(temp3)
        temp_pos_truth.append(temp4)
        temp_neg_truth.append(temp5)
        # temp_neg.append(temp6)
        # temp_pos.append(temp7)
    model_err_array.append(np.median(np.array(temp_model)))
    pos_count_array.append(np.median(np.array(temp_pos_count)))
    neg_count_array.append(np.median(np.array(temp_neg_count)))
    pos_truth_array.append(np.median(np.array(temp_pos_truth)))
    neg_truth_array.append(np.median(np.array(temp_neg_truth)))


model_err_array=np.array(model_err_array)
pos_count_array=np.array(pos_count_array)
neg_count_array=np.array(neg_count_array)
pos_truth_array=np.array(pos_truth_array)
neg_truth_array=np.array(neg_truth_array)
# false_pos_array=np.array(false_pos_array)
# false_neg_array=np.array(false_neg_array)
# np.gradient(total_err_array)
#
# plt.plot(np.arange(5000,60000,1000),gradient)
# plt.title('Derivative of Total Error Rate as Initial BF occupies 0.9-1.0 of Memory vs. Memory Space')
# plt.xlabel('Memory Space')
# plt.ylabel('Derivative of Total Error Rate Curve')
#======================================================
# """For Multi subplots"""
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharey=True,figsize=(15,6))
# fig.suptitle('How is Total Error Rate of the Learned Bloom Filter Change Regards Error Rate of the Different Sections',size=20)
# ax1.plot(np.arange(0,1.0,0.01),false_pos_array,linestyle='solid', ms=20,label='False Positive Rate')
# ax1.plot(np.arange(0,1.0,0.01),total_err_array, linestyle='dashdot', ms=20, label='Total Error Rate')
# ax2.plot(np.arange(0,1.0,0.01),false_neg_array, linestyle='solid', ms=20, label='False Negative Rate')
# ax2.plot(np.arange(0,1.0,0.01),total_err_array, linestyle='dashdot', ms=20, label='Total Error Rate')
# ax3.plot(np.arange(0,1.0,0.01),model_err_array, linestyle='solid', ms=20, label='Model Error Rate')
# ax3.plot(np.arange(0,1.0,0.01),total_err_array, linestyle='dashdot', ms=20, label='Total Error Rate')
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
# #plt.savefig('result1.pdf')
# plt.show()
#=================================================================
"""For multi lines plots"""
fig, ax = plt.subplots()
#declare a counter to filter out different sample classes
index=0
#for loop to plot grouped plot
# ax.plot(np.arange(0,1.0,0.01),false_neg_array,linestyle='solid', ms=15,label='False Negative Rate')
# ax.plot(np.arange(0,1.0,0.01),false_pos_array,linestyle='solid', ms=15,label='False Positive Rate')

ax.plot(np.arange(0.1,0.9,0.01),model_err_array, linestyle='dotted', ms=15, label='Model Error Rate')
#ax.plot(np.arange(0,1.0,0.01),backup_err_array, linestyle='solid', ms=15, label='Back Up Filter Error Rate')
# ax.plot(np.arange(0,1.0,0.01),total_err_array, linestyle='dashdot', ms=15, label='Total Error Rate')
#ax.plot(np.arange(0,1.0,0.01),err_result_array, linestyle='dashdot', ms=15, label='Initial Filter Error Rate')
# ax.set_xlabel('Threshold Value')
ax.set_ylabel('Model Error Rate')
ax.legend(loc=6,fontsize=8)
ax2=ax.twinx()
ax2.plot(np.arange(0.1,0.9,0.01),pos_count_array, linestyle='dashdot', ms=15, label='Predicted Positive Count')
ax2.plot(np.arange(0.1,0.9,0.01),neg_count_array, linestyle='dashdot', ms=15, label='Predicted Negative Count')
ax2.plot(np.arange(0.1,0.9,0.01),pos_truth_array, linestyle='solid', ms=15, label='Real Positive Count')
ax2.plot(np.arange(0.1,0.9,0.01),neg_truth_array, linestyle='solid', ms=15, label='Real Negative Count')
ax2.set_ylabel('Count')
ax2.legend(loc=7,fontsize=8)
# ax.plot(np.arange(5000,20000,500),gradient, linestyle='dashdot', ms=15)
plt.xlabel('Threshold Value')
# plt.ylabel('Error Rate')
plt.title('Changes in Counts as Threshold Value Chnages--6000 Memory')
plt.savefig('result.pdf')
plt.show()
#=========================================================
"""For Grouped Dot Plots"""
# import scipy.optimize as optimize
# def curve_func(x,a,b,c):
#    return a*np.exp(-b*x)-c
# popt, pcov = optimize.curve_fit(curve_func, np.arange(0.99,1.0,0.0001), total_err_array,maxfev=100000)
# fitted_curve = curve_func(np.arange(0.99,1.0,0.0001), *popt)
fig, ax = plt.subplots()
#declare a counter to filter out different sample classes
index=0
#for loop to plot grouped plot
for num in set(hash_result_array):
    ax.plot(np.arange(0,1.0,0.01)[index:index+len(hash_result_array[hash_result_array==num])]
    ,total_err_array[index:index+len(hash_result_array[hash_result_array==num])],
            marker='o', linestyle='', ms=8,label='# of Hash '+str(num))
    index=index+len(hash_result_array[hash_result_array==num])
#ax.plot(np.arange(0.99,1.0,0.0001),fitted_curve,linestyle='-',ms=8,label='Fitted Curve')
ax.legend()
plt.xlabel('% of Memory for Initial BF')
plt.ylabel('Error Rate')
plt.title('% of Memory for Initial BF from 0 to 1.0 with Interval 0.01 with # of Hash Functions--10000 Memory')
plt.legend(loc=1,fontsize=6)
#plt.savefig('rouhded_result.pdf')
plt.show()
