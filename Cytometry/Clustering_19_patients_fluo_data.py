# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:06:06 2020

@author: eliej
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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


n= 100000 #Number of Samples
#PAT1
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


#Join 3 dataframes

dfs = [df2, df3,df4,df5,df6,df7,df72,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19]
df_keys = pd.concat(dfs)
feat_cols = df_keys.columns[1:-13]
class_cols3 = df_keys.columns[-13:]


#TAg humancells
df2['Patient'] = 1
df3['Patient'] = 2
df4['Patient'] = 3
df5['Patient'] = 4
df6['Patient'] = 5
df7['Patient'] = 6
df72['Patient'] = 7
df8['Patient'] = 8
df9['Patient'] = 9
df10['Patient'] = 10
df11['Patient'] = 11
df12['Patient'] = 12
df13['Patient'] = 13
df14['Patient'] = 14
df15['Patient'] = 15
df16['Patient'] = 16
df17['Patient'] = 17
df18['Patient'] = 18
df19['Patient'] = 19


dfpat = [df2, df3,df4,df5,df6,df7,df72,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19]
df_pat = pd.concat(dfpat)
df_patpat = df_pat.loc[df_pat['is_cd4']==1,]


#Clusters
df = df_keys.loc[df_keys['is_cd4']==1,]
sampling=df[feat_cols]
#real_sampling=sampling[['FITC-A','PerCP-Cy5-5-A', 'APC-A', 'Alexa Fluor 700-A', 'APC-Cy7-A', 'PE-YG-A','PE-Texas Red-A', 'PE-Cy5-A', 'PE-Cy7-A', 'Pacific Blue-A','Horizon V500-A', 'Brillant Violet 605-A']]
real_sampling=sampling[['APC-Cy7-A','PerCP-Cy5-5-A','Brillant Violet 605-A','Alexa Fluor 700-A','PE-Cy5-A','PE-Texas Red-A','Pacific Blue-A','FITC-A','PE-YG-A','PE-Cy7-A']]
real_sampling=real_sampling.apply(zscore)
technique=anndata.AnnData(real_sampling)
adata=technique

#sc.pp.log1p(adata)
adata.raw = adata
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)

sc.pp.neighbors(adata, n_neighbors= 10, n_pcs=7)
sc.tl.umap(adata)
sc.pl.umap(adata)

umap_coordinates = adata.obsm['X_umap'] 
kmeans = KMeans(n_clusters=3,random_state=0).fit(umap_coordinates) 
adata.obs['kmeans'] = kmeans.labels_ 
adata.obs['kmeans'] = adata.obs['kmeans'].astype(str)
sc.pl.umap(adata, color='kmeans')

sc.tl.louvain(adata, resolution=0.1)
sc.pl.umap(adata, color='louvain')
adata.obs['Patient'] = df_patpat['Patient'].values

sc.pl.umap(adata, color=['APC-Cy7-A','PerCP-Cy5-5-A','Brillant Violet 605-A','Alexa Fluor 700-A','PE-Cy5-A','PE-Texas Red-A','Pacific Blue-A','FITC-A','PE-YG-A','PE-Cy7-A','louvain'], vmin=-3,vmax=3)
sc.pl.umap(adata, color= 'Patient')






























