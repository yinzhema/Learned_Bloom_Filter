import pandas as pd
import numpy as np
import tensorflow as tf

class Model_Wrapper:
    """This class is a wrapper class for Tesorflow model to be fed into SLBF or ALBF"""
    def  __init__(self,model):
        self.model=model

    def Predict(self,x):
        interpreter = tf.lite.Interpreter(model_content=self.model)
        interpreter.allocate_tensors()
        #print(x)
        if len(x.shape)==1:
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'],np.array(x).astype(np.float32).reshape((1,x.shape[0])))
            interpreter.invoke()
            return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0]

        else:
            result=[]
            for i in range(x.shape[0]):
                interpreter.set_tensor(interpreter.get_input_details()[0]['index'],np.array(x.iloc[i,:]).astype(np.float32).reshape((1,x.shape[1])))
                interpreter.invoke()
                result.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0])
            return result
