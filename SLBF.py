import Bloom_filters as BF
import pandas as pd
import numpy as np
import tensorflow as tf

class SLBF:
    def  __init__(self):
        """set variables as placeholder for initial and backup filter"""
        self.initial=''
        self.backup=''
        self.model=''
        self.threshold=0

    def construct(self, dataset, label_column_name, url_column_name,model,model_dataset, memory_size,backup_size,threshold):
        label=np.array(dataset[label_column_name])
        all_data=np.array(dataset[url_column_name])
        try:
            all_pos_data=np.array(dataset[dataset[label_column_name]==1][url_column_name])
        except:
            return "Need a ground truth column set to 1/0"

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

        """Classifier"""
        interpreter = tf.lite.Interpreter(model_content=model)
        interpreter.allocate_tensors()
        self.model=interpreter
        self.threshold=threshold

        positive_array=[]
        for i in round1_pos:
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'],np.array(model_dataset.iloc[i,:-1]).astype(np.float32).reshape((1,model_dataset.shape[1]-1)))
            interpreter.invoke()
            positive_array.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0])
        #the data instance that are positive from initial filter and prob of 1 is bigger than T
        positive_array=(np.array(positive_array)>=threshold)

        #creating backup filter and label variables
        backup_filter_data=all_data[round1_pos][positive_array!=True]
        backup_filter_label=label[round1_pos][positive_array!=True]
        pos_backup_filter_data=[]
        for i in range(len(backup_filter_data)):
            if backup_filter_label[i]==True:
                pos_backup_filter_data.append(backup_filter_data[i])

        """Backup Filter"""
        backup_bf=BF.Bloom_Filter(backup_size,len(pos_backup_filter_data))
        for i in range(len(pos_backup_filter_data)):
            backup_bf.insert(pos_backup_filter_data[i])

        #run data through the backup filter
        backup_bf_result=[]
        backup_bf_truth=[]
        for i in range(len(backup_filter_data)):
            backup_bf_result.append(backup_bf.search(backup_filter_data[i]))
            backup_bf_truth.append(backup_filter_label[i])

        self.backup=backup_bf

        return
    def search(self,data,model_data):

        if self.initial.search(data):
            self.model.set_tensor(self.model.get_input_details()[0]['index'],np.array(model_data[:-1]).astype(np.float32).reshape((1,len(model_data)-1)))
            self.model.invoke()
            if self.model.get_tensor(self.model.get_output_details()[0]['index'])[0][0]>self.threshold:
                return True
            else:
                if self.backup.search(data):
                    return True
                else:
                    return False
        else:
            return False
