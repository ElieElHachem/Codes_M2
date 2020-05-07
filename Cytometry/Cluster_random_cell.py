# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:44:23 2020

@author: eliej
"""

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


os.chdir('C:\\Users\\eliej\\Desktop\\Documents_folder_stage\\Documents\\Cytometry\\Patients\\Patient_random')

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
sample=combined_csv.iloc[:,0:3]
real_sampling=sample.apply(zscore)
adata=anndata.AnnData(real_sampling)

adata.raw = adata
sc.tl.pca(adata, svd_solver='arpack')
pca_coordinates = adata.obsm['X_pca'] 
kmeans = KMeans(n_clusters=3,random_state=0).fit(pca_coordinates) 
adata.obs['kmeans'] = kmeans.labels_ 
adata.obs['kmeans'] = adata.obs['kmeans'].astype(str)

adata.obs['Patient'] = combined_csv['Patient_Phénotype'].values

sc.pp.neighbors(adata, n_neighbors= 10, n_pcs=7)
sc.tl.leiden(adata, resolution=0.1)
sc.tl.umap(adata)
sc.pl.pca(adata, color= ['kmeans','leiden','Patient'])


umap_coordinates = adata.obsm['X_umap'] 
kmeans = KMeans(n_clusters=3,random_state=0).fit(umap_coordinates) 
adata.obs['umap_kmeans'] = kmeans.labels_ 
adata.obs['umap_kmeans'] = adata.obs['umap_kmeans'].astype(str)

adata.obs['Patient'] = combined_csv['Patient_Phénotype'].values

sc.pl.umap(adata, color=['umap_kmeans','leiden','Patient'])



