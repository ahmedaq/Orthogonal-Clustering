% OC_Gaussian_fixed - Function for executing the complex-valued MMSE estimate of
% a Gaussian distributed sparse signal using orthoganal clustering approach
% where "equal length semi-orthogonal clusters are used".
%
% This function implements the algorithm described in "Structure-Based
% Bayesian Sparse Reconstruction", Ahmed A. Quadeer & Tareq Y. Al-Naffouri.
% OC_Gaussian_fixed will find a MMSE estimate of sparse solutions "x"
% for the underdetermined system of linear equations
%                           yp = Psi*x + n,
% where n is complex Gaussian additive noise, 
%       x is Gaussian distributed, and
%       Psi is the partial DFT measurement matrix 
%
% SYNTAX: xmmse_final = OC_Gaussian_fixed(yp,Psi,sigma_imp,N0,pp,index,L)
%
% Outputs:  xmmse_final = The MMSE estimate of the sparse vector
%
% Inputs:   yp = n x 1 observation vector
%           Psi = n x m measurement/sensing matrix (partial DFT matrix)
%           sigma_imp = Non-zero elements' variance in the sparse vector
%           N0 = Noise variance
%           pp = Probability of non-zero elements occurence
%           index = "n-length" vector representing indices of continuous
%           sensing columns
%           L = Length of the clusters (e.g., 4,8,16,32,...)
%
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

function xmmse_final = ...
    OC_Gaussian_fixed(yp,Psi,sigma_imp,N0,pp,index,L)

%% Initialization
sig2s = [0; sigma_imp]; %Sparse signal variance
sig2w=N0;               %Noise variance

%% Step 1: Determine dominant positions using correlation

[n,m] = size(Psi);     %Size of Psi

yp_corr = Psi'*yp;     %Performing correlation

%% Step 2: Form equal length semi-orthogonal clusters

[~,b1] = sort(abs(yp_corr),'descend');

P = min(n, 1 + ceil(m*pp + erfcinv(1e-1)*sqrt(2*m*pp*(1 - pp))));  % Approx. no. of clusters to be formed

s1 = zeros(P,1);
J_cluster = cell(P,1);
kk = 1;

for mm = 1:m
    s1(kk) = b1(mm);
    if kk == 1
        J_cluster{kk} = mod(s1(kk)-L/2 : s1(kk)+L/2-1,m);
        J_cluster{kk}([J_cluster{kk}] == 0) = m;
        kk = kk+1;
    else
        if sum(ismember(mod(s1(kk)-L/2:s1(kk)+L/2-1,m),[J_cluster{:}]))==0 %check if any member of the cluster to be made is present in previous clusters
            J_cluster{kk} = mod(s1(kk)-L/2 : s1(kk)+L/2-1,m);
            J_cluster{kk}([J_cluster{kk}] == 0) = m;
            kk = kk+1;
        end
    end
    if kk > P
        break
    end
end
    
%Re-ordering sort_b11 according to the clusters (as clusters are not in
%correct order after we joined first n last cluster)
temp = [];
for kk = 1:length(J_cluster)
    temp = [temp J_cluster{kk}];
end
sort_b11 = temp;

%% Step 3: Find the dominant supports and their likelihoods (Main algorithm)

%Initialization
D = L;
P_c = 2;                %Search length within a cluster
AA = Psi;
A = AA(:,sort_b11(1:L));
ps = [1 - pp; pp];

T = cell(P,1);          %Indices of active taps
nu = cell(P,1);         %Likelihood values
xmmse = cell(P,1);      %Initial MMSE estimate of x
Omega = cell(D,P_c);
Xi = cell(D,P_c);
xmmse_p = zeros(m,P);   %MMSE estimate of all clusters
nup = zeros(2*L+1,P);   %Vector consisting of all best likelihoods 
p_nup = zeros(2*L+1,P); %Probability of best likelihoods

for p = 1:P             %Loop for each cluster p
    
    T{p} = cell(D,P_c);          %Indices of active taps in cluster p
    nu{p} = -inf*ones(D,P_c);    %Likelihood values in cluster p
    xmmse{p} = cell(D,P_c);      %MMSE estimate of cluster p
    
    if p == 1                    %If first cluster
        Omega_root = A/sig2w;
        Xi_root = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A).*Omega_root)).^(-1));
        z = yp;
    else                         %If subsequent clusters
        dd = mod(J_cluster{p}(1)-J_cluster{1}(1),m);        %Finding the difference delta between cluster p and cluster 1
        wd = (exp(-sqrt(-1)*(2*pi)/m*(index(1:n)-1)*dd)).'; %Calculating the modulation vector
        z = yp.*conj(wd);                                   %Modulation
    end
    
    nu_root = -norm(z)^2/sig2w + L*log(ps(1));  %Root node
    nuxt_root = zeros(1,L);	
    
    nuxt_root(1:L) = nu_root + log(Xi_root/sig2s(2)) + ...
        Xi_root.*abs(z'*Omega_root).^2 + log(ps(2)/ps(1));           
    
    p_c = 1;
    
    for d = 1:D
        
        nstar = d;
        T{p}{d,p_c} = J_cluster{p}(nstar);
        nu{p}(d,p_c) = nuxt_root(nstar);
        if p == 1   %Only need to update Omega and Xi for first cluster. Remaining clusters use the same values
            Omega{d,p_c} = Omega_root; %Omega{1,:} will all be equal to Omega_root
            Xi{d,p_c} = Xi_root;
            Omega{d,p_c+1} = Omega{d,p_c} - Omega{d,p_c}(:,nstar)*Xi{d,p_c}(nstar)*( Omega{d,p_c}(:,nstar)'*A );
            Xi{d,p_c+1} = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A).*Omega{d,p_c+1})).^(-1));
        end
        xmmse{p}{d,p_c} = zeros(m,1);
        xmmse{p}{d,p_c}(T{p}{d,p_c}) = sig2s(2)*Omega{d,p_c+1}(:,nstar)'*z;
        
%         nuxt = zeros(1,L);
        nuxt(1:L) = nu{p}(d,p_c) + log(Xi{d,p_c+1}/sig2s(2)) ...
            + Xi{d,p_c+1}.*abs(z'*Omega{d,p_c+1}).^2 + log(ps(2)/ps(1)); 
        % can't activate an already activated coefficient!
        nuxt(nstar) = -inf*ones(size(T{p}{d,p_c}));
        [nustar,nqstar] = max(nuxt);                        %Find best extension
        while sum(abs(nustar-nu{p}(1:d-1,p_c+1)) < 1e-8)    %If same as explored node...
            nuxt(nqstar) = -inf;                            %Mark extension as redundant
            [nustar, nqstar] = max(nuxt);                   %Find next best extension
        end
        nstar2 = mod(nqstar - 1, m) + 1;                    %Index of best extension
        nu{p}(d,p_c+1) = nustar;                            %Replace worst explored node...
        T{p}{d,p_c+1} = [T{p}{d,p_c}, J_cluster{p}(nstar2)];

        Omegatt = Omega{d,p_c+1} - Omega{d,p_c+1}(:,nstar2)*Xi{d,p_c+1}(nstar2)*( Omega{d,p_c+1}(:,nstar2)'*A); %AA(:,sort_b11((p-1)*L+1:p*L))
        Xitt = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A).*Omegatt)).^(-1));

        xmmse{p}{d,p_c+1} = zeros(m,1);
        xmmse{p}{d,p_c+1}(T{p}{d,p_c+1}) = sig2s(2)*Omegatt(:,[nstar,nstar2])'*z;       

    end
    
    %% Step 4: Evaluate the estimate of x
    
    nup(:,p) = [nu_root ; nu{p}(:)]; %including nu_root to include the case when no impulse is present in cluster
    p_nup(:,p) = exp(nup(:,p)-max(nup(:,p)))/sum(exp(nup(:,p)-max(nup(:,p))));
    xmmse_p(:,p) = [zeros(m,1) xmmse{p}{:}]*p_nup(:,p);
end

xmmse_final = sum(xmmse_p,2);