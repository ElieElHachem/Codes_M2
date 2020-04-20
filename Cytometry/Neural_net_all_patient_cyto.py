#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:47:52 2020

@author: elie
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import zscore
import keras.backend as K
min_max_scaler = preprocessing.MinMaxScaler()


""" Now lets optimized this neural network for only 7 variables """
df2 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_001_new.csv')
df3 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_002_new.csv')
df4 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_003_new.csv')
df5 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_004_new.csv')
df6 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_005_new.csv')
df7 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_006_new.csv')
df72 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_007_new.csv')
df8 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_008_new.csv')
df9 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_009_new.csv')
df10 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_010_new.csv')
df11 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_011_new.csv')
df12 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_012_new.csv')
df13 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_013_new.csv')
df14 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_014_new.csv')
df15 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_015_new.csv')
df16 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_016_new.csv')
df17 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_017_new.csv')
df19 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_019_new.csv')



dfs = [df2, df3,df4,df5,df6,df7,df72,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df19]
df0 = pd.concat(dfs)
#df0 = df0.loc[df0['is_CD8']==1,]


           #For CD4 and CD8      
feat_cols = df0.columns[1:-13]
class_cols = df0.columns[-9:]
sampling=df0[feat_cols]
real_sampling=sampling[['APC-Cy7-A','PerCP-Cy5-5-A','Brillant Violet 605-A','Alexa Fluor 700-A','PE-Cy5-A','PE-Texas Red-A','Pacific Blue-A','FITC-A','PE-YG-A','PE-Cy7-A','Horizon V500-A','APC-A']]
real_sampling=real_sampling.apply(zscore)
real_sampling=real_sampling.values
sampling2=df0[class_cols].values

           #Lets Train the Neural Net
                         
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(real_sampling)
X_scale

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, sampling2, test_size=0.3)

model = Sequential()
model.add(Dense(5000, activation='relu', input_shape=(12,)))
model.add(Dropout(0.3))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(9, activation='sigmoid')) #sigmoid

sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
adam= Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy', 
              optimizer=adam , metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=500, epochs=100,
          validation_data=(X_val_and_test, Y_val_and_test), shuffle=True)

model.save("neural_net_optimiedsoft_max_adam_binary_crossentropy_100_10mark_9variables_all_patient.h5")
model=load_model("neural_net_optimiedsoft_max_adam_binary_crossentropy_100_10mark_9variables_all_patient.h5")
        
        #Test the model for 1 patient
df02 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_018_new.csv')
#df02 = df02.loc[df02['is_CD8']==1,]
#dfx2 = df02.loc[df02['is_CD8']==1,]
#dfx00=[dfx1, dfx2]
#df02=pd.concat(dfx00)

feat_cols = df02.columns[1:-13]
class_cols = df02.columns[-9:]
sampling2=df02[feat_cols]
real_sampling2=sampling2[['APC-Cy7-A','PerCP-Cy5-5-A','Brillant Violet 605-A','Alexa Fluor 700-A','PE-Cy5-A','PE-Texas Red-A','Pacific Blue-A','FITC-A','PE-YG-A','PE-Cy7-A','Horizon V500-A','APC-A']]
real_sampling2=real_sampling2.apply(zscore)
real_sampling2=real_sampling2.values
sampling02=df02[class_cols].values

X_new = min_max_scaler.fit_transform(real_sampling2)
y_pred=model.predict(X_new)
threshold= 0.5
y_pred[y_pred >= threshold] = 1
y_pred[y_pred < threshold] = 0
u=model.evaluate(X_new, y_pred)[1]

zz=(sampling02 == y_pred)
ff=sum(sampling02 == y_pred)
uu=ff/len(sampling02)  #accuracy for each variables
np.mean(uu) #General accuracy

temp = pd.DataFrame((sampling02 == y_pred))
calc=pd.DataFrame.sum(temp, axis=1)
zz=np.where(calc==9)
len(zz[0])/len(sampling02) #Accurate classification on 7 combined values

















































# model.evaluate(X_val_and_test, Y_val_and_test)[1]
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()
# plt.figure()
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()

# plt.figure()
# plt.plot(hist.history['f1_metric'])
# plt.plot(hist.history['val_f1_metric'])
# plt.title('F1 accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()
