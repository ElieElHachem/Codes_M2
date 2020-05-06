# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:13:14 2020

@author: eliej
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#On génère la première loi normale et donc la première colonne de données

mu_type=[2,3,8,10,8,10.4,9] #pour les differentes lois normales
sigma_type=[1,1,1,1,1,1,1]  #Eccartype pour les differentes lois normales
number_of_cells=500

data_first_column = np.random.normal(loc=mu_type[0], scale=sigma_type[0], size=number_of_cells)
#On génère la seconde colonne de donnée avec 2 gausiennes (car 2 possibilitées possibles)
seuil1=0.8
second_column =[]
for data in data_first_column:
    if data >seuil1:
        r=random.gauss(mu_type[1], sigma_type[1])
        second_column.append(r)
    else:
        mu3=8
        sigma3=1
        r=random.gauss(mu_type[2],sigma_type[2])
        second_column.append(r)
        
#Fusions des listes
data_tuples = list(zip(data_first_column,second_column))
data2_column=pd.DataFrame(data_tuples, columns=['First_column','Second_column'])
#On génère la 3ème colonne de donnée avec 4 gausiennes (4 posibilitées)
seuil2=14 #on choisi un seuil 2
third_column =[]
for i in range(len(data2_column)):
    if data2_column.loc[i,'First_column'] > seuil1 and data2_column.loc[i,'Second_column']>seuil2:
        #Patient 1-10 & 13- 15= 10 / Patient 11-13 = 23
        r=random.gauss(mu_type[3],sigma_type[3])
        third_column.append(r)
    elif data2_column.loc[i,'First_column'] > seuil1 and data2_column.loc[i,'Second_column']<seuil2:
        #Patient 1-13= 8 / Patient 14-16 = 11
        r=random.gauss(mu_type[4],sigma_type[4])
        third_column.append(r)
    elif data2_column.loc[i,'First_column'] < seuil1 and data2_column.loc[i,'Second_column']>seuil2:
        #Patient 1-16= 7 / Patient 17-20 = 10.4
        r=random.gauss(mu_type[5],sigma_type[5])
        third_column.append(r)
    else:
        r=random.gauss(mu_type[6],sigma_type[6])
        third_column.append(r)
#Dataframe finale
data2_column["Third_column"]=third_column
data2_column.to_csv ('Patient_21_random.csv', index = False, header=True)


