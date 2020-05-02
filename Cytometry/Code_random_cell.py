# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:13:14 2020

@author: eliej
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

#On génère la première loi normale et donc la première colonne de données
mu = 2
sigma = 1.0
data_first_column = np.random.randn(300) * sigma + mu

#On génère la seconde colonne de donnée avec 2 gausiennes (car 2 possibilitées possibles)

seuil1=0.8
second_column =[]
for data in data_first_column:
    if data >seuil1:
        mu2=3
        sigma2=1
        r=randint(12, 14)*sigma2 + mu2
        second_column.append(r)
    else:
        mu3=8
        sigma3=1
        r=randint(12, 14) *sigma3 + mu3
        second_column.append(r)
        
#Fusions des listes
data_tuples = list(zip(data_first_column,second_column))
data2_column=pd.DataFrame(data_tuples, columns=['First_column','Second_column'])

#On génère la 3ème colonne de donnée avec 4 gausiennes (4 posibilitées)
seuil2=14 #on choisi un seuil 2
third_column =[]
for i in range(len(data2_column)):
    if data2_column.loc[i,'First_column'] > seuil1 and data2_column.loc[i,'Second_column']>seuil2:
        mu4=10
        sigma4=1
        r=randint(12, 14)*sigma4 + mu4
        third_column.append(r)
    elif data2_column.loc[i,'First_column'] > seuil1 and data2_column.loc[i,'Second_column']<seuil2:
        mu5=8
        sigma5=1
        r=randint(12, 14)*sigma5 + mu5
        third_column.append(r)
    elif data2_column.loc[i,'First_column'] < seuil1 and data2_column.loc[i,'Second_column']>seuil2:
        mu6=7
        sigma6=1
        r=randint(12, 14)*sigma6 + mu6
        third_column.append(r)
    else:
        mu7=9
        sigma7=1
        r=randint(12, 14)*sigma7 + mu7
        third_column.append(r)

#Dataframe finale
data2_column["Third_column"]=third_column

