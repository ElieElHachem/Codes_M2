# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 01:55:15 2020

@author: eliej
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import glob
import scipy
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import math as mt
from scipy.stats import variation 
from sklearn.decomposition import PCA 
import gseapy
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import scanpy as sc
from copy import deepcopy
import matplotlib.pyplot as plt
import anndata
from scipy.sparse import csr_matrix
from scipy.stats import zscore


#Code basique 1 patient
df2 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_001_new.csv')
feat_cols = df2.columns[1:-13]
class_cols3 = df2.columns[-9:]

sampling02=df2[class_cols3].values
sampling02 = pd.DataFrame(sampling02, dtype='int')

sampling02['Merged'] = sampling02[sampling02.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)   #On extrait les 9 colonnes qu'on regroupe dans une même colonne appelée Merged

uu = sampling02['Merged'].tolist() #On extrait la colonne merged, qu'on transforme en list dans uu

nouvlist=[] #Ici la list  uu était en str, du coup on la transforme en int.
for i in uu:
    z=[int(a) for a in str(i)]
    nouvlist.append(z)
    
    
list2= np.array(nouvlist) #On transforme la list en multiple array like
result=[] #On store les valeurs pour chaque celule dans result
#hist = np.zeros(list2.size)  #J'ai testé avec ça, mais le problème c'est que ça donne une matrice de 194895x1 uniquement
for member in range(len(list2)): 
    b_num = list2[member]
    value = 0
    value  = int(b_num.dot(2**np.arange(b_num.size)[::-1]))
    #hist[value] += 1
    result.append(value) #Du coup on stock une valeurpour chaque cellule dans une colonne.
  
number_of_cell=len(list2)  
newrez=  result
newrez[:] = [x / number_of_cell for x in newrez] #On fait le ratio en fonction du nombre de cellules
  
hehe = pd.DataFrame(newrez) 
hist = hehe.hist() #Plot de l'histogramme





#Code 19 patiens

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
df18 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_018_new.csv')
df19 = pd.read_csv('C:/Users/eliej/Desktop/Documents_folder_stage/Documents/FCS_reading/labeld_specimen_019_new.csv')

dfs =dfs = [df2, df3,df4,df5,df6,df7,df72,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19]
df_keys = pd.concat(dfs)
feat_cols = df_keys.columns[1:-13]
class_cols3 = df_keys.columns[-9:]

sampling02=df_keys[class_cols3].values
sampling02 = pd.DataFrame(sampling02, dtype='int')

sampling02['Merged'] = sampling02[sampling02.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)   #On extrait les 9 colonnes qu'on regroupe dans une même colonne appelée Merged

uu = sampling02['Merged'].tolist() #On extrait la colonne merged, qu'on transforme en list dans uu

nouvlist=[] #Ici la list  uu était en str, du coup on la transforme en int.
for i in uu:
    z=[int(a) for a in str(i)]
    nouvlist.append(z)
    
    
list2= np.array(nouvlist) #On transforme la list en multiple array like
result=[] #On store les valeurs pour chaque celule dans result
#hist = np.zeros(list2.size)  #J'ai testé avec ça, mais le problème c'est que ça donne une matrice de 194895x1 uniquement
for member in range(len(list2)): 
    b_num = list2[member]
    value = 0
    value  = int(b_num.dot(2**np.arange(b_num.size)[::-1]))
    #hist[value] += 1
    result.append(value) #Du coup on stock une valeurpour chaque cellule dans une colonne.
  
number_of_cell=len(list2)  
newrez=  result
newrez[:] = [x / number_of_cell for x in newrez] #On fait le ratio en fonction du nombre de cellules
  
hehe = pd.DataFrame(newrez) 
hist = hehe.hist() #Plot de l'histogramme



