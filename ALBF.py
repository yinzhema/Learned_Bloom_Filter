import Bloom_filters as BF
import Adaptive_Bloom_filters as ABF
import pandas as pd
import numpy as np
import tensorflow as tf

class ALBF:
    def  __init__(self):
        """set variables as placeholder for initial and backup filter"""
        self.initial=''
        self.backup=''
        self.model=''
        self.cutoff={}

    def construct(self, dataset, label_column_name, url_column_name,model,model_dataset, memory_size,backup_size,c,seg,sandwich=True):

        """Initialize all the variables"""
        try:
            all_pos_data=np.array(dataset[dataset[label_column_name]==1][url_column_name])
        except:
            return "Need a ground truth column set to 1/0"

        all_data=np.array(dataset[url_column_name])
        label=np.array(dataset[label_column_name])
        dataset['index']=dataset.index

        if sandwich:
            """Initial Filter"""
            initial_bf=BF.Bloom_Filter(memory_size-backup_size,len(all_pos_data))
            for i in range(len(all_pos_data)):
                initial_bf.insert(all_pos_data[i])

            #run the data through initial bloom filter
            round1_pos=[]
            for i in range(len(all_data)):
                if initial_bf.search(all_data[i])==True:
                    round1_pos.append(i)

            self.initial=initial_bf

            backup_bf=ABF.Bloom_Filter(backup_size,len(round1_pos),seg)

        else:

            round1_pos=np.arange(len(all_data))
            initial_err=0
            backup_bf=ABF.Bloom_Filter(backup_size,len(round1_pos),seg)

        """creating data frame for backup adaptive filter"""
        round1_pos_df=dataset.iloc[round1_pos,:]

        """Classifier"""
        self.model=model
        #self.threshold=threshold
        round1_pos_df=model_dataset.iloc[round1_pos,:].astype(np.float32)#.reshape((1,model_dataset.shape[1]-1))
        # interpreter = tf.lite.Interpreter(model_content=model)
        # interpreter.allocate_tensors()
        # self.model=interpreter
        #
        # model_output=[]
        # for i in round1_pos:
        #     interpreter.set_tensor(interpreter.get_input_details()[0]['index'],np.array(model_dataset.iloc[i,:-1]).astype(np.float32).reshape((1,model_dataset.shape[1]-1)))
        #     interpreter.invoke()
        #     model_output.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0])
        model_output=self.model.Predict(model_dataset.iloc[round1_pos,:-1].astype(np.float32))
        round1_pos_df['Prob']=model_output

        round1_pos_df=round1_pos_df.sort_values(by='Prob').reset_index()

        """Backup ABF"""
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
                self.cutoff[round1_pos_df.loc[m,'Prob']]=j
            else:
                round1_pos_df.loc[m1:m,'Segments']=j
                self.cutoff[round1_pos_df.loc[m1,'Prob']]=j
            m0=int(m0*(1-c))
            m1=m
            m=m+m0

        #filter out the automatic positive: the points in the last segment
        positive_array=round1_pos_df[round1_pos_df['Segments']==(backup_bf.segments-1)]['index'].tolist()

        #Store the rest of dataframe that is not auotmatic postive in backup_bf_df
        backup_bf_df=round1_pos_df[round1_pos_df['Segments']!=(backup_bf.segments-1)]

        #print(backup_bf_df)
        #Those backup points that are 1
        pos_backup_bf_df=backup_bf_df[backup_bf_df[label_column_name]==1]

        #set the index of backup_bf_df and pos_backup_bf_df to index
        backup_bf_df.set_index('index',inplace=True)
        pos_backup_bf_df.set_index('index',inplace=True)

        #store them in the backup adaptive bloom filter
        for i in pos_backup_bf_df.index.tolist():
            backup_bf.insert(all_data[i],pos_backup_bf_df.loc[i,'Segments'])

        #Checking all backup dataset using the ABF
        backup_bf_result=[]
        backup_bf_truth=[]
        for i in backup_bf_df.index.tolist():
            backup_bf_result.append(backup_bf.search(all_data[i],backup_bf_df.loc[i,'Segments']))
            backup_bf_truth.append(int(label[i]))

        self.backup=backup_bf

    def search(self,data,model_data):
        if self.initial=='':
            # self.model.set_tensor(self.model.get_input_details()[0]['index'],np.array(model_data[:-1]).astype(np.float32).reshape((1,len(model_data)-1)))
            # self.model.invoke()
            prob=self.model.Predict(model_data[:-1])
            for i in self.cutoff:
                if prob>i:
                    segment=self.cutoff[i]
                    if segment==self.backup.segments-1:
                        return True
                    else:
                        if self.backup.search(data,segment):
                            return True
                        else:
                            return False
        else:
            if self.initial.search(data):
                # self.model.set_tensor(self.model.get_input_details()[0]['index'],np.array(model_data[:-1]).astype(np.float32).reshape((1,len(model_data)-1)))
                # self.model.invoke()
                prob=self.model.Predict(model_data[:-1])
                for i in self.cutoff:
                    if prob>i:
                        segment=self.cutoff[i]
                        if segment==self.backup.segments-1:
                            return True
                        else:
                            if self.backup.search(data,segment):
                                return True
                            else:
                                return False
            else:
                return False
