# stDGAC: a novel identifying spatial domains method via graph attention contrastive network for spatial transcriptomics data

## Overview

stDGAC is a model used to identify spatial domains by jointly using a denoising autoencoder and a graph attention contrastive network for subsequent analysis of spatial transcriptomic data. The pre-trained denoising autoencoder performed dimensionality reduction and denoising, and then the low-dimensional latent representation is learned through a graph attention contrastive network to aggregate the neighborhood information of the spatial context and acquire a more robust and discriminative feature representation. The reconstructed expression profile can be used for downstream analysis such as spatial domain recognition, trajectory inference and gene expression data denoising.

![stDGAC](https://github.com/JING-ING/stDGAC/blob/main/stDGAC.jpg)

## Requirements

Python packages:

* python==3.9
* anndata==0.7.6
* torch>=1.11.0
* cudnn>=10.2
* h5py==3.7.0
* scanpy==1.9.3
* numpy==1.21.6
* pandas==1.5.0
* scanpy==1.9.3
* scipy==1.8.1
* scikit-learn==1.3.0

## Installation

cd stDGAC

python setup.py build

python setup.py install
