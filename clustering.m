% clustering - Clustering based on threshold
% Function to construct semi-orthogonal clusters based on threshold 
%
% Inputs:   indx_gt_thresh = vector consisting of indices greater than
%                            threshold
%           m = Length of sparse signal
%
% Output:   J_cluster = Cells consisting of clusters
%           cluster_indices = Vector consisting of all indices in the
%                             clusters arranged according to J_cluster
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 18, 2012
% BLC_Gaussian version 1.0
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012



function [J_cluster,cluster_indices] = clustering(indx_gt_thresh,m)

mmm = 1;
for mm = 1:length(indx_gt_thresh)
    b11 = indx_gt_thresh(mm);
    b11_new(mmm:1:mmm+4)=mod([b11-2 b11-1 b11 b11+1 b11+2],m);
    ind = b11_new == 0;
    b11_new(ind) = m;
    mmm = mmm+5;
    cluster_indices = unique(b11_new);
    diff_b11 = [cluster_indices(2:end) 0]-cluster_indices;
end

%Joining the clusters if very close
kk = 1;
len_cluster = 0;
for ss = 1:length(diff_b11)
    if (diff_b11(ss) == 1 || diff_b11(ss) == 2 || diff_b11(ss) == 3 || diff_b11(ss) == 4)% || diff_b11(ss) == 5)
        len_cluster = len_cluster+1;
    else
        J_cluster{kk} = cluster_indices(ss-len_cluster:ss);
        kk = kk+1;
        len_cluster = 0;
    end
end

%Joining the first and the last cluster if required
if J_cluster{1}(1) == 1 && J_cluster{end}(end) == m && length(J_cluster{1})~=m
    J_cluster{end} = [J_cluster{end} J_cluster{1}];
    J_cluster{1} = [];
end

%Removing the empty cells
J_cluster(cellfun(@isempty,J_cluster))=[]; 

%Making sure that the indices in cluster are from 1 to m
for kk = 1:length(J_cluster)
    J_cluster{kk} = mod(J_cluster{kk}(1):1:J_cluster{kk}(1)+length(J_cluster{kk})-1,m);
    J_cluster{kk}([J_cluster{kk}] == 0) = m;
end

D = [cellfun('length',J_cluster)].';    %Finding length of each cluster
[~,cluster_max_len] = max(D);           %Finding the cluster of max. length

%Swapping the cluster with max. length with the first cluster
dummy = J_cluster{1};
J_cluster{1} = J_cluster{cluster_max_len};
J_cluster{cluster_max_len} = dummy;

%Constructing "cluster_indices"
temp = [];
for kk = 1:length(J_cluster)
    temp = [temp J_cluster{kk}];
end
cluster_indices = temp;