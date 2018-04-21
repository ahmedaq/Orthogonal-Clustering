% clustering_seq - Sequential Clustering
% Function to construct semi-orthogonal clusters sequentially 
%
% Inputs:   yp_corr = Correlation values
%           p = Probability of non-zero elements occurence
%           m = Length of sparse signal
%           n = Number of sensing columns
%
% Output:   J_cluster = Cells consisting of clusters
%           cluster_indices = Vector consisting of all indices in the
%                             clusters arranged according to J_cluster
%           P = Number of clusters formed
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 18, 2012
% BLC_Gaussian version 1.0
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012


function [J_cluster,cluster_indices,P] = clustering_seq(yp_corr,p,m,n)

[~,b1] = sort(abs(yp_corr),'descend');
P = min(n, ceil(m*p + erfcinv(0.1)*sqrt(2*m*p*(1 - p))));  % Approx. no. of clusters to form

mmm = 1;
for mm = 1:m
    b11 = b1(mm);
    b11_new(mmm:1:mmm+4)=mod([b11-2 b11-1 b11 b11+1 b11+2],m);
    ind = b11_new == 0;
    b11_new(ind) = m;
    mmm = mmm+5;
    cluster_indices = unique(b11_new);
    diff_b11 = [cluster_indices(2:end) 1]-cluster_indices;
    no_of_clusters = length(find(diff_b11~=1)) - length(find(diff_b11==2)) ...
        - length(find(diff_b11==3)) - length(find(diff_b11==-m));
    if no_of_clusters > P
        break;
    end
end

kk = 1;
len_cluster = 0;
for ss = 1:length(diff_b11)
    if (diff_b11(ss) == 1 || diff_b11(ss) == 2 || diff_b11(ss) == 3)% || diff_b11(ss) == 4)
        len_cluster = len_cluster+1;
    else
        J_cluster{kk} = cluster_indices(ss-len_cluster:ss);
        len_cluster_each(kk) = length(J_cluster{kk});
        kk = kk+1;
        len_cluster = 0;
    end
end

if J_cluster{1}(1) == 1 && J_cluster{end}(end) == m && length(J_cluster{1})~=m
    J_cluster{end} = [J_cluster{end} J_cluster{1}];
    J_cluster{1} = [];
end
J_cluster(cellfun(@isempty,J_cluster))=[]; %Removing the empty cells

for kk = 1:length(J_cluster)
    J_cluster{kk} = mod(J_cluster{kk}(1):1:J_cluster{kk}(1)+length(J_cluster{kk})-1,m);
    J_cluster{kk}([J_cluster{kk}] == 0) = m;
end

len_clusters = [cellfun('length',J_cluster)].'; %finding length of each cluster
[~,cluster_max_len] = max(len_clusters);  %finding the cluster of max. length
%Swapping the cluster with max. length with the first cluster
dummy = J_cluster{1};
J_cluster{1} = J_cluster{cluster_max_len};
J_cluster{cluster_max_len} = dummy;

%Constructing cluster_indices
temp = [];
for kk = 1:length(J_cluster)
    temp = [temp J_cluster{kk}];
end
cluster_indices = temp;