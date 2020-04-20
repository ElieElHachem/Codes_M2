#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:24:26 2020

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


""" This is the neural network optimized for 13 variables
"""

df2 = pd.read_csv('/home/elie/Documents/Cytometry/Patients/patient_2/labeled_specimen_002.csv')
df3 = pd.read_csv('/home/elie/Documents/Cytometry/Patients/patient_3/labeled_specimen_003.csv')

dfs = [df2, df3]
df0 = pd.concat(dfs)
df=df0.iloc[:,0:18]
df=df.apply(zscore)
dff=df.values
feat_cols = dff[:,0:18]
dff2=df0.values
class_cols = dff2[:,20:34]

X_scale = min_max_scaler.fit_transform(feat_cols)
X_scale

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, class_cols, test_size=0.4)
#
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model = Sequential()
model.add(Dense(5000, activation='relu', input_shape=(18,)))
model.add(Dropout(0.1))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(13, activation='sigmoid'))

sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
adam= Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=adam , metrics=['accuracy',f1_metric])

hist = model.fit(X_train, Y_train,
          batch_size=100000, epochs=1000,
          validation_data=(X_val_and_test, Y_val_and_test), shuffle=True)

model.save("neural_net_optimied_adam_binary_epoch1000.h5")
model=load_model("neural_net_optimied_adam_binary_epoch1000.h5", custom_objects={'f1_metric': f1_metric})
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


df02 = pd.read_csv('/home/elie/Documents/Cytometry/Patients/patient_1/labeld_specimen_001.csv')
df2=df02.iloc[:,1:19]
df2=df2.apply(zscore)
dff=df2.values
feat_cols = dff[:,0:18]
dff2=df02.values
class_cols = dff2[:,21:34]

X_new = min_max_scaler.fit_transform(feat_cols)
y_pred=model.predict(X_new)
threshold= 0.5
y_pred[y_pred >= threshold] = 1
y_pred[y_pred < threshold] = 0
u=model.evaluate(X_new, y_pred)[1]


ff=sum(class_cols == y_pred)
uu=ff/len(class_cols)  #accuracy for each variables
np.mean(uu) #General accuracy

temp = pd.DataFrame((class_cols == y_pred))
calc=pd.DataFrame.sum(temp, axis=1)
zz=np.where(calc==13)
len(zz[0])/len(class_cols) #Accurate classification on 13 combined values

print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')


""" Now lets optimized this neural network for only 7 variables """
df2 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_001_new.csv')
df3 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_002_new.csv')
df4 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_003_new.csv')
df5 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_004_new.csv')
df6 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_005_new.csv')
df7 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_006_new.csv')
df8 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_007_new.csv')

dfs = [df2, df3,df4,df5,df6,df7,df8]
df0 = pd.concat(dfs)


           #For CD4 and CD8      
feat_cols = df0.columns[1:-13]
class_cols = df0.columns[-7:]
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
model.add(Dense(50000, activation='relu', input_shape=(12,)))
model.add(Dropout(0.3))
model.add(Dense(6000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(7, activation='sigmoid'))

sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
adam= Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=adam , metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=1500, epochs=100,
          validation_data=(X_val_and_test, Y_val_and_test), shuffle=True)

model.save("neural_net_optimied_adam_binary_epoch100_10mark_7variables_cd4_and_cd8.h5")
model=load_model("neural_net_optimied_adam_binary_epoch100_10mark_7variables_cd4_and_cd8.h5")
        
        #Test the model for 1 patient
df02 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_008_new.csv')
#dfx1 = df02.loc[df02['is_cd4']==1,]
#dfx2 = df02.loc[df02['is_CD8']==1,]
#dfx00=[dfx1, dfx2]
#df02=pd.concat(dfx00)

feat_cols = df02.columns[1:-13]
class_cols = df02.columns[-7:]
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
zz=np.where(calc==7)
len(zz[0])/len(sampling02) #Accurate classification on 7 combined values

print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')

""" Now lets optimized this neural network for only 7 variables for CD4 only """
df2 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_001_new.csv')
df3 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_002_new.csv')
df4 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_003_new.csv')
df5 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_004_new.csv')
df6 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_005_new.csv')
df7 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_006_new.csv')
df8 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_007_new.csv')

dfs = [df2, df3,df4,df5,df6,df7,df8]
df0 = pd.concat(dfs)
df0 = df0.loc[df0['is_cd4']==1,]


           #For CD4
feat_cols = df0.columns[1:-12]
class_cols = df0.columns[-7:]
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
model.add(Dense(50000, activation='relu', input_shape=(12,)))
model.add(Dropout(0.3))
model.add(Dense(6000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(7, activation='sigmoid'))

sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
adam= Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=adam , metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=50000, epochs=100,
          validation_data=(X_val_and_test, Y_val_and_test), shuffle=True)

model.save("neural_net_optimied_adam_binary_epoch100_12mark_7variables_cd4_only.h5")
model=load_model("neural_net_optimied_adam_binary_epoch100_12mark_7variables_cd4_only.h5")

        #Test the model for 1 patient
        
df02 = pd.read_csv('/home/elie/Documents/FCS_reading/labeld_specimen_008_new.csv')
df02 = df02.loc[df02['is_cd4']==1,]

feat_cols = df02.columns[1:-13]
class_cols = df02.columns[-7:]
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
u=model.evaluate(X_new, y_pred)

zz=(sampling02 == y_pred)
ff=sum(sampling02 == y_pred)
uu=ff/len(sampling02)  #accuracy for each variables
np.mean(uu) #General accuracy

temp = pd.DataFrame((sampling02 == y_pred))
calc=pd.DataFrame.sum(temp, axis=1)
zz=np.where(calc==7)
len(zz[0])/len(sampling02) #Accurate classification on 7 combined values

print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')

""" Now lets optimized this neural network for only 7 variables for CD8 only """
df2 = pd.read_csv('/home/elie/Documents/Cytometry/Patients/patient_2/labeled_specimen_002.csv')
df3 = pd.read_csv('/home/elie/Documents/Cytometry/Patients/patient_3/labeled_specimen_003.csv')
dfs = [df2, df3]
df0 = pd.concat(dfs)
df0 = df0.loc[df0['is_cd8']==1,]


           #For CD8      
feat_cols = df0.columns[1:-13]
class_cols = df0.columns[-7:]
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
model.add(Dense(500, activation='relu', input_shape=(12,)))
model.add(Dropout(0.1))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(7, activation='sigmoid'))

sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
adam= Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=adam , metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=70000, epochs=100,
          validation_data=(X_val_and_test, Y_val_and_test), shuffle=True)

model.save("neural_net_optimied_adam_binary_epoch100_10mark_7variables_cd8_only.h5")
model = load_model('neural_net_optimied_adam_binary_epoch100_10mark_7variables_cd8_only.h5')

        #Test the model for 1 patient
        
df02 = pd.read_csv('/home/elie/Documents/Cytometry/Patients/patient_1/labeld_specimen_001.csv')
df02 = df02.loc[df02['is_cd8']==1,]

feat_cols = df02.columns[1:-13]
class_cols = df02.columns[-7:]
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
zz=np.where(calc==7)
len(zz[0])/len(sampling02) #Accurate classification on 7 combined values




















