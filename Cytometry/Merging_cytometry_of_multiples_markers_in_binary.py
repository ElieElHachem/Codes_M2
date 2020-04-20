#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:39:11 2020

@author: elie
"""

import fcsparser
import pandas as pd
import numpy as np


#This is for Patient 1
    #Upload data lympho
path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/Lymphocyte.fcs'
meta, datalympho = fcsparser.parse(path, reformat_meta=True)
index=list(datalympho.columns)
datalympho['is_lympho'] = 1
    #Upload data CD3
path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD3.fcs'
meta, datacd3 = fcsparser.parse(path, reformat_meta=True)
datacd3['is_cd3'] = 1
    #Merge lympho and CD3
result001 = pd.merge(datalympho, datacd3, how='left' ,on= index)

#This is for Patient 2
    #Upload CD4
path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4.fcs'
meta, datacd4 = fcsparser.parse(path, reformat_meta=True)
datacd4['is_cd4'] = 1
result002 = pd.merge(result001, datacd4, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_CCR7.fcs'
meta, datacd4_ccr7 = fcsparser.parse(path, reformat_meta=True)
datacd4_ccr7['is_ccr7'] = 1
result003 = pd.merge(result002, datacd4_ccr7, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_CD27.fcs'
meta, CD4_CD27 = fcsparser.parse(path, reformat_meta=True)
CD4_CD27['is_cd27'] = 1
result004 = pd.merge(result003, CD4_CD27, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_CD28.fcs'
meta, CD4_CD28 = fcsparser.parse(path, reformat_meta=True)
CD4_CD28['is_cd28'] = 1
result005 = pd.merge(result004, CD4_CD28, how='left' ,on= index)


path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_CD45RA.fcs'
meta, CD4_CD45RA = fcsparser.parse(path, reformat_meta=True)
CD4_CD45RA['is_cd45ra'] = 1
result006 = pd.merge(result005, CD4_CD45RA, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_CD57.fcs'
meta, CD4_CD57 = fcsparser.parse(path, reformat_meta=True)
CD4_CD57['is_cd57'] = 1
result007 = pd.merge(result006, CD4_CD57, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_CD279.fcs'
meta, CD4_CD279 = fcsparser.parse(path, reformat_meta=True)
CD4_CD279['is_cd279'] = 1
result008 = pd.merge(result007, CD4_CD279, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD4_KLRG1.fcs'
meta, CD4_KLRG1 = fcsparser.parse(path, reformat_meta=True)
CD4_KLRG1['is_KLRG1'] = 1
result009 = pd.merge(result008, CD4_KLRG1, how='left' ,on= index)

index2=list(result009.columns)

#Upload CD8

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8.fcs'
meta, CD8 = fcsparser.parse(path, reformat_meta=True)
CD8['is_CD8'] = 1
result010 = pd.merge(result009, CD8, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_CCR7.fcs'
meta, CD8_CCR7 = fcsparser.parse(path, reformat_meta=True)
CD8_CCR7['is_ccr7'] = 1
result011 = pd.merge(result010, CD8_CCR7, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_CD27.fcs'
meta, CD8_CD27 = fcsparser.parse(path, reformat_meta=True)
CD8_CD27['is_cd27'] = 1
result012 = pd.merge(result011, CD8_CD27, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_CD28.fcs'
meta, CD8_CD28 = fcsparser.parse(path, reformat_meta=True)
CD8_CD28['is_cd28'] = 1
result0121 = pd.merge(result012, CD8_CD28, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_CD45RA.fcs'
meta, CD8_CD45RA = fcsparser.parse(path, reformat_meta=True)
CD8_CD45RA['is_cd45ra'] = 1
result013 = pd.merge(result0121, CD8_CD45RA, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_CD57.fcs'
meta, CD8_CD57 = fcsparser.parse(path, reformat_meta=True)
CD8_CD57['is_cd57'] = 1
result014 = pd.merge(result013, CD8_CD57, how='left' ,on= index)


path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_CD279.fcs'
meta, CD8_CD279 = fcsparser.parse(path, reformat_meta=True)
CD8_CD279['is_cd279'] = 1
result015 = pd.merge(result014, CD8_CD279, how='left' ,on= index)

path = '/home/elie/Desktop/FACS_data/Patient_files/Patient_16/CD8_KLRG1.fcs'
meta, CD8_KLRG1 = fcsparser.parse(path, reformat_meta=True)
CD8_KLRG1['is_KLRG1'] = 1
result016 = pd.merge(result015, CD8_KLRG1, how='left' ,on= index)


last_result=result016.fillna(0)
last_result['is_ccr7']=last_result['is_ccr7_x']+last_result['is_ccr7_y']
last_result['is_cd27']=last_result['is_cd27_x']+last_result['is_cd27_y']
last_result['is_cd28']=last_result['is_cd28_x']+last_result['is_cd28_y']
last_result['is_cd45ra']=last_result['is_cd45ra_x']+last_result['is_cd45ra_y']
last_result['is_cd57']=last_result['is_cd57_x']+last_result['is_cd57_y']
last_result['is_cd279']=last_result['is_cd279_x']+last_result['is_cd279_y']
last_result['is_KLRG1']=last_result['is_KLRG1_x']+last_result['is_KLRG1_y']
index2.append('is_CD8')
index2[23],index2[30] = index2[30], index2[23]
real_sampling=last_result[index2]


real_sampling.to_csv ('labeld_specimen_016_new.csv', index = False, header=True)





