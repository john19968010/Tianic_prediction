#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import urllib.request 
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt


# In[ ]:


def downloadData(url,file):
    urllib.request.urlretrieve(url,file)
    data = pd.read_excel(file)
    return data


# In[ ]:


def PreprocessData(data):
    draft = data.drop(['name'], axis=1)
    age_avg = draft['age'].mean()
    draft['age'] = draft['age'].fillna(age_avg)
    fare_avg = draft['fare'].mean()
    draft['fare'] = draft['fare'].fillna(fare_avg)
    draft['sex'] = draft['sex'].map({'female':0, 'male':1}).astype(int)
    new_data = pd.get_dummies(data=draft,columns=['embarked'])
    
    New_data = new_data.values
    Label = New_data[:,0]
    Features = New_data[:,1:]
    
    scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    Scaled_Features = scale.fit_transform(Features)
    
    return Scaled_Features,Label
    


# In[ ]:


def splitData(data):
    split = np.random.rand(len(data)) < 0.7
    train = data[split]
    test = data[~split]
    return train,test


# In[ ]:


def traing_model():
    model = Sequential()
    model.add(Dense(units=40, input_dim=9,
                kernel_initializer='uniform',
                activation='relu'))
    model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


def training_result(train_Feature,train_Label,model):
    train_history = model.fit(x=train_Feature,
                          y=train_Label,
                          validation_split=0.1,
                          epochs=30,
                          batch_size=30,verbose=2)
    plt.plot(train_history.history[accuracy])
    plt.plot(train_history.history[val_accuracy])
    plt.title('Train History')
    plt.xlabel('訓練週期')
    plt.ylabel(train)
    plt.legend([accuracy,val_accuracy], loc='upper left')
    plt.show()
    


# In[ ]:


def scores(model,test_Feature,test_Label):
    scores = model.evaluate(x=test_Feature,
                         y=test_Label)
    return scores[1] 


# In[ ]:


#執行處
url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
file = "titanic3.xls"
data = downloadData(url,file)
train,test = splitData(data)
train_Feature,train_Label = PreprocessData(train)
test_Feature,test_Label = PreprocessData(test)
model = traing_model()
result = training_result(train_Feature,train_Label,model)
point = scores(model,test_Feature,test_Label)

