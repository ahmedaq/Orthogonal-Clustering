# Orthogonal-Clustering (OC) method

## Table of Contents
*  [Overview](#overview)
*  [Details](#details)
*  [Requirements](#requirements)
*  [Usage](#usage)
*  [Troubleshooting](#troubleshooting)

## Overview
A Bayesian method which utilises the rich structure embedded in the sensing matrix for fast sparse signal recovery

## Details
#### Title of paper
Structure-based Bayesian sparse reconstruction
#### Authors
Ahmed A. Quadeer and Tareq Y. Al-Naffouri

## Requirements
1. A Windows, Unix/Linux, Macintosh machine capable of running MathWorks MATLAB software version R2011b or later. The software package may work on previous releases of MATLAB, but it was not checked for compatibility.

2. In order to compare the performance of OC with other sparse signal reconstruction algorithms, one would need to download the packages from their respective websites:
    * Fast Bayesian Matching Pursuit (FBMP) http://www2.ece.ohio-state.edu/~zinielj/fbmp/download.html.
    * Orthogonal Matching Pursuit (OMP) http://www.personal.soton.ac.uk/tb1m08/sparsify/sparsify.html.

## Usage
#### Effect of the sparsity rate "p" on the performance of the OC method and comparison with other sparse reconstruction algorithms for the case when the sparse signal is Gaussian distributed.
   * Run the script ```Experiment_Effect_p_Gaussian.m```
#### Effect of the signal-to-noise ratio "SNR" on the performance of the OC method and comparison with other sparse reconstruction algorithms for the case when the sparse signal is Gaussian distributed.
   * Run the script ```Experiment_Effect_SNR_Gaussian.m```
#### Effect of the cluster length "L" on the performance of the OC method
   * Run the script ```Experiment_Effect_L.m```
#### Effect of the under-sampling ratio "us" on the performance of the OC method
   * Run the script ```Experiment_Effect_Undersampling_ratio.m```
#### Effect of the sparsity rate "p" on the performance of the OC method and comparison with other sparse reconstruction algorithms for the case when the sparse signal is non-Gaussian distributed.
   * Run the script ```Experiment_Effect_p_nonGaussian.m```
#### Effect of the signal-to-noise ratio "SNR" on the performance of the OC method and comparison with other sparse reconstruction algorithms for the case when the sparse signal is non-Gaussian distributed.
   * Run the script ```Experiment_Effect_SNR_nonGaussian.m```
   
## Troubleshooting
For any questions or comments, please email at ahmedaq@gmail.com. 

## Citation
#### Plain text
Quadeer, Ahmed A. & Al-Naffouri, T. Y. Structure-Based Bayesian Sparse Reconstruction. IEEE Trans. Signal Process. 60, 6354–6367 (2012).
#### BibTeX
@article{Quadeer2012,
author = {Quadeer, Ahmed A. and Al-Naffouri, T. Y.},
doi = {10.1109/TSP.2012.2215029},
issn = {1053-587X},
journal = {IEEE Trans. Signal Process.},
keywords = {Compressive Sensing,MyPublications},
month = {dec},
number = {12},
pages = {6354--6367},
title = {{Structure-Based Bayesian Sparse Reconstruction}},
url = {http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6280684},
volume = {60},
year = {2012}
}
