"""this file is used for comparing the performance of SLBF/LBF/ALBF
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


"""Adaptive Bloom Filter"""
def learned_adaptive_filter(mem,backup,c,seg,sandwich=True):
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

"""LBF"""
def learned_filter(mem,backup,t):
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
    backup_bf=BF.Bloom_Filter(backup,len(pos_backup_filter_data))

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
    #=============================================================================
    #0.6185 is calculated based on false pos rate=p^k=p^(ln2*(m/2)) where p is % of bits that are 0 after hashed and when p=0.5, p^ln2=0.6185
    #
    ideal_total_err=(0.6185**(initial_bf.array_size/initial_bf.length))*(model_false_pos+(1-model_false_pos)*(0.6185**((backup_bf.array_size/backup_bf.length))))

    return total_err#backup_bf.false_pos,initial_err,initial_bf.false_pos,backup_err,backup_bf.false_pos, total_err,ideal_total_err

def experiment(mem):
    error1=learned_filter(mem,2500,0.6)
    c=np.arange(0.1,0.9,0.1)
    backup=np.arange(1000,mem,500)
    error2=0
    error3=0
    temp_error=[]
    for i in c:
        temp=[]
        for j in backup:
            temp.append(learned_adaptive_filter(mem,j,i,10))
        temp_error.append([learned_adaptive_filter(0,mem,i,10,sandwich=False),np.amin(np.array(temp))])
    temp_error=np.array(temp_error).T
    error2=np.amin(temp_error[0])
    error3=np.amin(temp_error[1])
    return [error1,error2,error3]

if __name__ == "__main__" :
     p=multiprocessing.Pool(processes=6)
     total_error=p.map(experiment,[i for i in np.arange(8000,18000,500)])
     p.close()


total_error=np.array(total_error).T

label=['SLBF','ALBF','SALBF']
fig, ax = plt.subplots()
for i in range(len(total_error)):
    ax.plot(np.arange(8000,18000,500),total_error[i], linestyle='dotted', ms=15, label='Data Structure: '+str(label[i]))
ax.set_xlabel('Memory Size')
ax.set_ylabel('False Positive Rate')
ax.legend(loc=1,fontsize=8)
plt.title('Memory Size vs. False Positive Rate with Varied Data Structure')
plt.savefig('result.pdf')
plt.show()
