#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:35:32 2020

@author: elie
"""

import scipy
import scvelo as scv
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


#IMPORT DATA FROM THE FIRST EXPERIMENT ON THE MICE (P12 mice)
filesample= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Merrick_et_al/SC_murinep12/GSM3717977_SCmurinep12_matrix.mtx'
col_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Merrick_et_al/SC_murinep12/GSM3717977_SCmurinep12_barcodes.tsv'
row_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Merrick_et_al/SC_murinep12/GSM3717977_SCmurinep12_genes.tsv'
col_names=pd.read_csv(col_names, sep='\t', header= None)
row_names=pd.read_csv(row_names, sep='\t', header= None)
onlysymbol=pd.DataFrame(row_names.iloc[:,1])
CMatrix=scipy.io.mmread(filesample)
sc.settings.verbosity = 3             
Tmatrix_data=csr_matrix((np.transpose(CMatrix))).toarray()
uu=pd.DataFrame(Tmatrix_data,index=col_names.iloc[:,0])
uu.columns = onlysymbol.iloc[:,0]
technique=anndata.AnnData(uu)
adata_ref=technique #Setting Murinep12 as reference


#IMPORT DATA FROM THE SECOND EXPRIMENT ON THE MICE (ADULT MICE)
filesample= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Merrick_et_al/SC_murineAdult/GSM3717978_SCmurineAdult_matrix.mtx'
col_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Merrick_et_al/SC_murineAdult/GSM3717978_SCmurineAdult_barcodes.tsv'
row_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Merrick_et_al/SC_murineAdult/GSM3717978_SCmurineAdult_genes.tsv'
col_names=pd.read_csv(col_names, sep='\t', header= None)
row_names=pd.read_csv(row_names, sep='\t', header= None)
onlysymbol=pd.DataFrame(row_names.iloc[:,1])
CMatrix=scipy.io.mmread(filesample)
sc.settings.verbosity = 3             
Tmatrix_data=csr_matrix((np.transpose(CMatrix))).toarray()
uu=pd.DataFrame(Tmatrix_data,index=col_names.iloc[:,0])
uu.columns = onlysymbol.iloc[:,0]
technique=anndata.AnnData(uu)
adata=technique


#Lets define the same variables
var_names = adata_ref.var_names.intersection(adata.var_names)
adata_ref = adata_ref[:, adata_ref.var_names.isin(var_names)]
adata = adata[:, adata.var_names.isin(var_names)] 

#Model Graph
#sc.pp.pca(adata_ref)
#sc.pp.neighbors(adata_ref)
#sc.tl.umap(adata_ref)
#sc.tl.louvain(adata_ref, resolution=0.1)
#sc.pl.umap(adata_ref, color='louvain')


#Mapping using Ingest
#sc.tl.ingest(adata, adata_ref, obs='louvain')
#adata.uns['louvain_colors'] = adata_ref.uns['louvain_colors']  # fix colors
#sc.pl.umap(adata, color=['louvain'], wspace=0.5)
adata_concat = adata_ref.concatenate(adata, batch_categories=['ref', 'new'])

#
#adata_concat.obs.louvain = adata_concat.obs.louvain.astype('category')
#adata_concat.obs.louvain.cat.reorder_categories(adata_ref.obs.louvain.cat.categories, inplace=True)  # fix category ordering
#adata_concat.uns['louvain_colors'] = adata_ref.uns['louvain_colors']
#sc.pl.umap(adata_concat, color=['batch', 'louvain'])

#Using BBKNN
sc.pp.neighbors(adata_concat)
sc.tl.pca(adata_concat)
sc.tl.louvain(adata_concat)
sc.external.pp.bbknn(adata_concat, batch_key='batch')  # running bbknn 1.3.6
sc.tl.umap(adata_concat)
sc.pl.umap(adata_concat, color=['batch', 'louvain'])
sc.pl.umap(adata_concat, color=['batch', 'louvain','Plin1','Fabp4','Adipoq','Pparg'])

#Import new ADATA from the third experience
filesample= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_gWAT_ShamCL_Linneg/matrix.mtx'
col_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_gWAT_ShamCL_Linneg/barcodes.tsv'
row_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_gWAT_ShamCL_Linneg/genes.tsv'
col_names=pd.read_csv(col_names, sep='\t', header= None)
row_names=pd.read_csv(row_names, sep='\t', header= None)
onlysymbol=pd.DataFrame(row_names.iloc[:,1])
CMatrix=scipy.io.mmread(filesample)
sc.settings.verbosity = 3             
Tmatrix_data=csr_matrix((np.transpose(CMatrix))).toarray()
uu=pd.DataFrame(Tmatrix_data,index=col_names.iloc[:,0])
uu.columns = onlysymbol.iloc[:,0]
technique=anndata.AnnData(uu)
adata=technique

#Lets define the same variables
var_names = adata_concat.var_names.intersection(adata.var_names)
adata_concat = adata_concat[:, adata_concat.var_names.isin(var_names)]
adata = adata[:, adata.var_names.isin(var_names)] 

#Model Graph
#sc.pp.pca(adata_concat)
#sc.pp.neighbors(adata_concat)
#sc.tl.umap(adata_concat)
#sc.tl.louvain(adata_concat, resolution=0.1)
#sc.pl.umap(adata_concat, color='louvain')


#Mapping using Ingest
#sc.tl.ingest(adata, adata_concat, obs='louvain')
#adata.uns['louvain_colors'] = adata_concat.uns['louvain_colors']  # fix colors
#sc.pl.umap(adata, color=['louvain'], wspace=0.5)
adata_concat2 = adata_concat.concatenate(adata, batch_categories=['ref', 'new'])

#BBKNN
sc.pp.neighbors(adata_concat2)
sc.tl.pca(adata_concat2)
sc.tl.louvain(adata_concat2)
sc.external.pp.bbknn(adata_concat2, batch_key='batch')  # running bbknn 1.3.6
sc.tl.umap(adata_concat2)
sc.pl.umap(adata_concat2, color=['batch', 'louvain'])
sc.pl.umap(adata_concat2, color=['batch', 'louvain','Plin1','Fabp4','Adipoq','Pparg'])


#Import new ADATA from the 4th experience
filesample= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Chealsea_et_al/GSM3034641_matrix.mtx'
col_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Chealsea_et_al/GSM3034641_barcodes.tsv'
row_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Chealsea_et_al/GSM3034641_genes.tsv'
col_names=pd.read_csv(col_names, sep='\t', header= None)
row_names=pd.read_csv(row_names, sep='\t', header= None)
onlysymbol=pd.DataFrame(row_names.iloc[:,1])
CMatrix=scipy.io.mmread(filesample)
sc.settings.verbosity = 3             
Tmatrix_data=csr_matrix((np.transpose(CMatrix))).toarray()
uu=pd.DataFrame(Tmatrix_data,index=col_names.iloc[:,0])
uu.columns = onlysymbol.iloc[:,0]
technique=anndata.AnnData(uu)
adata=technique

#Lets define the same variables
var_names = adata_concat2.var_names.intersection(adata.var_names)
adata_concat = adata_concat2[:, adata_concat2.var_names.isin(var_names)]
adata = adata[:, adata.var_names.isin(var_names)] 

adata_concat3 = adata_concat.concatenate(adata, batch_categories=['ref', 'new'])

#BBKNN
sc.pp.neighbors(adata_concat3)
sc.tl.pca(adata_concat3)
sc.tl.louvain(adata_concat3)
sc.external.pp.bbknn(adata_concat3, batch_key='batch')  # running bbknn 1.3.6
sc.tl.umap(adata_concat3)
sc.pl.umap(adata_concat3, color=['batch', 'louvain'])
sc.pl.umap(adata_concat3, color=['batch', 'louvain','Plin1','Fabp4','Adipoq','Pparg'])



#Import new ADATA from the 5th experience
filesample= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_gWAT_ShamCL_Linpos/matrix.mtx'
col_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_gWAT_ShamCL_Linpos/barcodes.tsv'
row_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_gWAT_ShamCL_Linpos/genes.tsv'
col_names=pd.read_csv(col_names, sep='\t', header= None)
row_names=pd.read_csv(row_names, sep='\t', header= None)
onlysymbol=pd.DataFrame(row_names.iloc[:,1])
CMatrix=scipy.io.mmread(filesample)
sc.settings.verbosity = 3             
Tmatrix_data=csr_matrix((np.transpose(CMatrix))).toarray()
uu=pd.DataFrame(Tmatrix_data,index=col_names.iloc[:,0])
uu.columns = onlysymbol.iloc[:,0]
technique=anndata.AnnData(uu)
adata=technique

#Lets define the same variables
var_names = adata_concat3.var_names.intersection(adata.var_names)
adata_concat = adata_concat3[:, adata_concat3.var_names.isin(var_names)]
adata = adata[:, adata.var_names.isin(var_names)] 
adata_concat4 = adata_concat.concatenate(adata, batch_categories=['ref', 'new'])

#BBKNN
sc.pp.neighbors(adata_concat4)
sc.tl.pca(adata_concat4)
sc.tl.louvain(adata_concat4)
sc.external.pp.bbknn(adata_concat4, batch_key='batch')  # running bbknn 1.3.6
sc.tl.umap(adata_concat4)
sc.pl.umap(adata_concat4, color=['batch', 'louvain'])
sc.pl.umap(adata_concat4, color=['batch', 'louvain','Plin1','Fabp4','Adipoq','Pparg'])


#Import new ADATA from the 6th experience
filesample= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_iWAT_ShalCK_3d_Linneg/matrix.mtx'
col_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_iWAT_ShalCK_3d_Linneg/barcodes.tsv'
row_names= '/home/elie/Documents/single_cell/Practice/Single_cell_data/Burl_et_al/AGG_iWAT_ShalCK_3d_Linneg/genes.tsv'
col_names=pd.read_csv(col_names, sep='\t', header= None)
row_names=pd.read_csv(row_names, sep='\t', header= None)
onlysymbol=pd.DataFrame(row_names.iloc[:,1])
CMatrix=scipy.io.mmread(filesample)
sc.settings.verbosity = 3             
Tmatrix_data=csr_matrix((np.transpose(CMatrix))).toarray()
uu=pd.DataFrame(Tmatrix_data,index=col_names.iloc[:,0])
uu.columns = onlysymbol.iloc[:,0]
technique=anndata.AnnData(uu)
adata=technique

#Lets define the same variables
var_names = adata_concat4.var_names.intersection(adata.var_names)
adata_concat = adata_concat4[:, adata_concat4.var_names.isin(var_names)]
adata = adata[:, adata.var_names.isin(var_names)] 
adata_concat5 = adata_concat.concatenate(adata, batch_categories=['ref', 'new'])

#BBKNN
sc.pp.neighbors(adata_concat5)
sc.tl.pca(adata_concat5)
sc.tl.louvain(adata_concat5)
sc.external.pp.bbknn(adata_concat5, batch_key='batch')  # running bbknn 1.3.6
sc.tl.umap(adata_concat5)
sc.pl.umap(adata_concat5, color=['batch', 'louvain'])
sc.pl.umap(adata_concat5, color=['batch', 'louvain','Plin1','Fabp4','Adipoq','Pparg'])
sc.pl.umap(adata_concat5, color=[ 'louvain'])
sc.pl.umap(adata_concat5, color=['Plin1','Fabp4','Adipoq','Pparg'])

adata_idenf = adata_concat5[adata_concat5.obs['louvain'].isin(['16']),:]
krkr=pd.DataFrame(adata_idenf.X, index=adata_idenf.obs_names,  columns=adata_idenf.var_names)
krkr.to_csv('Predipo_cluster16_combined.csv', header=True, index=True) 

sc.tl.umap(adata_idenf)
sc.tl.louvain(adata_idenf)
sc.pl.umap(adata_idenf, color=['Plin1','louvain','Fabp4','Adipoq','Pparg'])

