"""this file is for adaptive learned bloom filter using original combo dataset: spam email+url links
"""
import math
import numpy as np
import Adaptive_Bloom_filters as ABF
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
test_spam_df=pd.concat([spam_df.iloc[:1500,:],spam_df[spam_df[57]==0].iloc[:2000,:]]).reset_index().drop(['index'],axis=1)

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

def learned_filter(mem,backup,c,seg,sandwich=True):
    if sandwich:
        #Space reserve for the two bloom filters
        """Initial Filter"""
        initial_bf=BF.Bloom_Filter(mem-backup,len(all_pos_url))
        for i in range(len(all_pos_url)):
            initial_bf.insert(all_pos_url[i])

        #run the data through initial bloom filter
        round1_pos=[]
        for i in range(len(all_url)):
            if initial_bf.search(all_url[i])==True:
                round1_pos.append(i)
        #calculate the error rate of initial bf: False positive/All negatives
        initial_err=(len(round1_pos)-len(all_pos_url))/(len(all_url)-len(all_pos_url))

        #make k a parameter to tune-->in the ABF code
        backup_bf=ABF.Bloom_Filter(backup,len(round1_pos),seg)

    else:
        round1_pos=np.arange(len(all_url))
        initial_err=0
        backup_bf=ABF.Bloom_Filter(backup,len(round1_pos),seg)

    """Naive Bayes Classifier and Backup Adaptive Filter"""
    #create a backup filter based on the initial filter's positive outcome divided by 5

    #creating data frame for backup adaptive filter
    round1_pos_df=test_spam_df.loc[round1_pos,:]

    round1_pos_df['Prob']=probability_list[round1_pos]

    round1_pos_df=round1_pos_df.sort_values(by='Prob').reset_index()

    #calculate total c after going throuhgh n number of segments
    total_c=0
    for i in range(backup_bf.segments+1):
        total_c=total_c+((1-c)**i)

    #calculate the initial m: the number of points in the first segment, the segment with lowest prob
    m=int(len(round1_pos)/total_c)
    m0=m
    m1=0

    round1_pos_df['Segments']=0
    #for loop to assign each data point with a segment
    for j in range(backup_bf.segments):
        if j == (backup_bf.segments-1):
            round1_pos_df.loc[m:,'Segments']=j
        else:
            round1_pos_df.loc[m1:m,'Segments']=j
        m0=int(m0*(1-c))
        m1=m
        m=m+m0

    #filter out the automiatic positive: the points in the last segment
    positive_array=round1_pos_df[round1_pos_df['Segments']==backup_bf.segments-1]['index'].tolist()
    #calculate model error rate
    model_miss=len(positive_array)-sum(test_spam_df.iloc[positive_array,57])
    model_fp=(model_miss)/(len(round1_pos)-sum(test_spam_df.iloc[round1_pos,57]))
    #Store the rest of dataframe that is not auotmatic postive in backup_bf_df
    backup_bf_df=round1_pos_df[round1_pos_df['Segments']!=backup_bf.segments-1]
    #Those backup points that are 1
    pos_backup_bf_df=backup_bf_df[backup_bf_df[57]==1]
    #store them in the backup adaptive bloom filter
    for i in pos_backup_bf_df['index'].tolist():
        backup_bf.insert(all_url[i][1:],pos_backup_bf_df[pos_backup_bf_df['index']==i]['Segments'].tolist()[0])
    #Checking all backup dataset using the ABF
    backup_bf_result=[]
    backup_bf_truth=[]
    for i in backup_bf_df['index'].tolist():
        backup_bf_result.append(backup_bf.search(all_url[i][1:],backup_bf_df[backup_bf_df['index']==i]['Segments'].tolist()[0]))
        backup_bf_truth.append(int(all_url[i][0]))
    # print(backup_bf.fp_calculator())
    #print(backup_bf.segments)
    # print(backup_bf.distribution_dict)
    backup_miss=sum(backup_bf_result)-sum(np.array(backup_bf_truth)==1)

    backup_bf_error=backup_miss/sum(np.array(backup_bf_truth)==-1)

    total_error=(model_miss+backup_miss)/2000
    return total_error#initial_err,model_fp,backup_bf_error,total_error

learned_filter(12000,8000,0.3,8,sandwich=False)


def experiment(mem):
    error=[]
    k=np.arange(1000,mem,1000)
    c=np.arange(0.1,0.9,0.1)
    for i in k:
        temp_error=[]
        for j in c:
            temp_error.append(learned_filter(mem,i,j,10))
        error.append(np.amin(temp_error))
    return error

if __name__ == "__main__" :
     p=multiprocessing.Pool(processes=6)
     total_error=p.map(experiment,[i for i in np.arange(8000,18000,2000)])
     p.close()

total_error=np.array(total_error)

plt.plot(np.arange(0.1,0.9,0.05),total_error)
plt.xlabel('% of Initial BF Memory')
plt.ylabel('Total Error Rate')
plt.title('Initial BF Memory vs. Total Error Rate with Optimzed C Value')
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
#=================================================================
"""For multi lines plots"""
fig, ax = plt.subplots()
mem=np.arange(8000,18000,2000)
for i in range(len(total_error)):

    ax.plot(np.arange(1000,mem[i],1000),total_error[i], linestyle='dotted', ms=15, label='Memory Size: '+str(np.arange(8000,18000,2000)[i]))
    # min_index=np.argmin(total_error[i])
    # ax.plot(np.arange(1000,mem[i],1000)[min_index],total_error[i][min_index],marker="v",color="black")
    # loc=(math.log(np.average(model_fp[i])/((1-np.average(model_fp[i]))*((1/np.average(model_fn[i]))-1)))/math.log(0.6185))
    # ax.vlines(428*loc,ymin=0,ymax=0.2)

#ax.plot(np.arange(0.1,0.9,0.05),error, linestyle='solid', ms=15, label='Total Error Rate')
#ax.plot(np.arange(0,1.0,0.01),err_result_array, linestyle='dashdot', ms=15, label='Initial Filter Error Rate')
ax.set_xlabel('Backup Filter Memory')
ax.set_ylabel('False Positive Rate')
ax.legend(loc=1,fontsize=8)
# ax2=ax.twinx()
# ax2.plot(np.arange(0.1,0.9,0.05),c, linestyle='solid', ms=15, label='C Value',color='black')
# ax2.plot(np.arange(0.1,0.9,0.01),neg_count_array, linestyle='dashdot', ms=15, label='Predicted Negative Count')
# ax2.plot(np.arange(0.1,0.9,0.01),pos_truth_array, linestyle='solid', ms=15, label='Real Positive Count')
# ax2.plot(np.arange(0.1,0.9,0.01),neg_truth_array, linestyle='solid', ms=15, label='Real Negative Count')
# ax2.set_ylabel('C Value')
# ax2.legend(loc=7,fontsize=8)
# ax.plot(np.arange(5000,20000,500),gradient, linestyle='dashdot', ms=15)
#plt.xlabel('% of Initial BF Memory')
# plt.ylabel('Error Rate')
plt.title('Backup Filter Memory vs. False Positive Rate with Varied Memory Size (Optimized C)')
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
