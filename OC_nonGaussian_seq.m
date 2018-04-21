% OC_nonGaussian_seq - Function for executing the complex-valued MMSE estimate of
% a non-Gaussian distributed sparse signal using orthoganal clustering approach
% where "clusters are formed sequentially".
%
% This function implements the algorithm described in "Structure-Based
% Bayesian Sparse Reconstruction", Ahmed A. Quadeer & Tareq Y. Al-Naffouri.
% OC_nonGaussian_seq will find a MMSE estimate of sparse solutions "x"
% for the underdetermined system of linear equations
%                           yp = Psi*x + n,
% where n is complex Gaussian additive noise, 
%       x is non-Gaussian distributed, and
%       Psi is the partial DFT measurement matrix 
%
% SYNTAX:   xmmse_final = OC_nonGaussian_seq(yp, Psi, sigma_imp, N0, pp, index)
%
% Outputs:  xls_mmse_final = The MMSE estimate of the sparse vector
%           xls_map_final = The MAP estimate of the sparse vector
%
% Inputs:   yp = n x 1 observation vector
%           Psi = n x m measurement/sensing matrix (partial DFT matrix)
%           N0 = Noise variance
%           p = Probability of non-zero elements occurence
%           index = "n-length" vector representing indices of continuous
%           sensing columns
%
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% OC_Gaussian_seq version 1.0
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

function [xls_mmse_final,xls_map_final] = OC_nonGaussian_seq(y,Psi,N0,p,index)

%% Determine dominant positions using correlation
[n,m] = size(Psi);  %Size of Psi

yp_corr = Psi'*y;   %Performing correlation

%% Step 2: Form semi-orthogonal clusters using sequential method

[J_cluster,cluster_indices] = clustering_seq(yp_corr,p,m,n);

%% Step 3: Find the dominant supports and their likelihoods (Main
%% algorithm)

%Initialization
D = [cellfun('length',J_cluster)].';    %Finding length of each cluster
P = length(D);                          %Number of clusters
P_c = 2;                                %Initial P for cluster
A = Psi(:,cluster_indices(1:D(1)));     %Measurement matrix corresponding to first cluster     
p_c = 1;


T = cell(P,1);          %Indices of active taps
nu = cell(P,1);         %Likelihood
xls = cell(P,1);        %Initial LS estimate
xls_k_mmse = zeros(m,P);%MMSE estimate of each cluster k
xls_k_map = zeros(m,P); %MAP estimate of each cluster k

for k = 1:P                %Loop for each cluster k
    
    pp = p/m*D(k);         %Probability of non-zero signal in each cluster
    ps = [1 - pp; pp];
    
    T{k} = cell(D(k),P_c);          %Indices of active taps in cluster k
    nu{k} = -inf*ones(D(k),P_c);	%Likelihood in cluster k
    xls{k} = cell(D(k),P_c);        %Initial LS estimate of x in cluster k
    
    if k == 1;      %If first cluster
        z = y;
    else            %If subsequent clusters
        dd = mod(J_cluster{k}(1)-J_cluster{1}(1),m);        %Finding the difference delta between cluster p and cluster 1
        wd = (exp(-sqrt(-1)*(2*pi)/m*(index(1:n)-1)*dd)).'; %Calculating the modulation vector
        z = y.*conj(wd);                                    %Modulation
    end
    
    nu_root = -norm(z)^2/N0 ;    %Root node
    nuxt_root = (-norm(z)^2 + (abs(z'*A).^2))/N0 ; %Likelihood of support size 1
        
    for d = 1:D(k)          %Within a cluster
        nstar = d;
        T{k}{d,p_c} = J_cluster{k}(nstar);
        nu{k}(d,p_c) = nuxt_root(nstar);
        
        Lambda=1;   %As columns of psi are unit-norm
        psi = Psi(:,T{1}{d,p_c});
        if k == 1   %Only need to update eta, omega and xi for first cluster. Remaining clusters use the same values
            eta{d,p_c} = psi'*A;
            omega{d,p_c} = Lambda*eta{d,p_c};
            xi{d,p_c} = (1 - sum((conj(omega{d,p_c}).*eta{d,p_c}),1)).^(-1); %defining xi as inverse
        end
        xls{k}{d,p_c} = zeros(m,1);
        xls{k}{d,p_c}(T{k}{d,p_c}) = Lambda*psi'*z;
        
        alpha = z'*psi*omega{d,p_c}(1:D(k));
        beta = z'*A(:,1:D(k));
        
        nuxt = zeros(1,D(k));
        nuxt(1:D(k)) = nu{k}(d,p_c) + (xi{d,p_c}(1:D(k))/N0).*(abs(alpha).^2 ... %Likelihood of support size 2
            - 2*real(beta.*conj(alpha)) + abs(beta).^2) ;
        % can't activate an already activated coefficient
        nuxt(nstar) = -inf*ones(size(T{k}{d,p_c}));
        
        [nustar,nqstar] = max(nuxt);                        % find best extension
        while sum(abs(nustar-nu{k}(1:d-1,p_c+1)) < 1e-8)    % if same as explored node...
            nuxt(nqstar) = -inf;                            % ... mark extension as redundant
            [nustar, nqstar] = max(nuxt);                   % ... and find next best extension
        end
        nstar2 = mod(nqstar - 1, m) + 1;                    % coef index of best extension
        nu{k}(d,p_c+1) = nustar;                            % replace worst explored node...
        T{k}{d,p_c+1} = [T{k}{d,p_c}, J_cluster{k}(nstar2)];
        
        xls{k}{d,p_c+1} = zeros(m,1);
        xls{k}{d,p_c+1}(T{k}{d,p_c+1}) = [xls{k}{d,p_c}(T{k}{d,p_c}) + ...
            xi{d,p_c}(nstar2)*omega{d,p_c}(nstar2)*eta{d,p_c}(nstar2)'*xls{k}{d,p_c}(T{k}{d,p_c}) ...
            - xi{d,p_c}(nstar2)*omega{d,p_c}(nstar2)*A(:,nstar2)'*z ;...
            -xi{d,p_c}(nstar2)*eta{d,p_c}(nstar2)'*xls{k}{d,p_c}(T{k}{d,p_c}) + xi{d,p_c}(nstar2)*A(:,nstar2)'*z];
        
    end
        
    Pc = min(n, 1 + ceil(D(k)*p + erfcinv(1e-2)*sqrt(2*D(k)*p*(1 - p)))); %Dependent on actual p, not pp
    p_cc = 3;
    
    nnustar = zeros(1,Pc);  %Best Likelihood value of previous support size
    TT = cell(1,Pc);        %Actual supports index
    nxt = cell(1,Pc);       %New likelihood values
    nnstar = cell(1,Pc);    %Best next support index in a cluster
    NNSTAR = cell(1,Pc);    %Best supports index in a cluster
    nnqstar = cell(1,Pc);   %Temporary storage of nnstar
    xxls = cell(1,Pc);
    
    [nnustar(p_cc-1),b] = max(nu{k}(:,2));
    TT{p_cc-1} = T{k}{b,2};
    nstar = find(J_cluster{k}==T{k}{b,2}(1));
    nstar2 = find(J_cluster{k}==T{k}{b,2}(2));
    NNSTAR{p_cc-1} = [nstar nstar2];
    xxls{p_cc-1} = xls{k}{b,2};
    
    while p_cc <= Pc
        psi2 = Psi(:,[TT{p_cc-1}]);
        Lambda2=(psi2'*psi2)^(-1);
        B = Psi(:,[J_cluster{k}]);
        eta2 = psi2'*B;
        omega2 = Lambda2*eta2;
        xi2 = (1 - sum((conj(omega2).*eta2),1)).^(-1); %defining xi as inverse
        alpha2 = y'*psi2*omega2;
        beta2 = y'*B;
        
        nxt{p_cc} = nnustar(p_cc-1) + (xi2/N0).*(abs(alpha2).^2 ...
            - 2*real(beta2.*conj(alpha2)) + abs(beta2).^2);
        nxt{p_cc}(NNSTAR{p_cc-1}) = -inf*ones(1,p_cc-1);
        [nnustar(p_cc),nnqstar{p_cc}] = max(real(nxt{p_cc}));
        nnstar{p_cc} = mod(nnqstar{p_cc} - 1, m) + 1;
        TT{p_cc} = [TT{p_cc-1}, J_cluster{k}([nnstar{p_cc}])];
        NNSTAR{p_cc} = [NNSTAR{p_cc-1} nnstar{p_cc}];
        
        xxls{p_cc} = zeros(m,1);
        xxls{p_cc}(TT{p_cc}) = [xxls{p_cc-1}(TT{p_cc-1}) + ...
            xi2(nnstar{p_cc})*omega2(:,nnstar{p_cc})*eta2(:,nnstar{p_cc})'*xxls{p_cc-1}(TT{p_cc-1}) ...
            - xi2(nnstar{p_cc})*omega2(:,nnstar{p_cc})*B(:,nnstar{p_cc})'*y ;...
            -xi2(nnstar{p_cc})*eta2(:,nnstar{p_cc})'*xxls{p_cc-1}(TT{p_cc-1}) + xi2(:,nnstar{p_cc})*B(:,nnstar{p_cc})'*y];
        
        p_cc = p_cc+1;
    end
    
    %% Step 4: Evaluate the estimate of x
    
    %MMSE
    p_of_S = [1:size([nu{k}],2) ;D(k)-(1:size([nu{k}],2))].'*[log(ps(2)) log(ps(1))].'; %p(S) of size (no. of supports size x 1)
    nu_p = [nu{k}]+repmat(p_of_S.',size([nu{k}]) - [0 1]);
    p_of_S_gt_3 = [1:length(nnustar) ;D(k)-(1:length(nnustar))].'*[log(ps(2)) log(ps(1))].';
    nup_gt_3 = nnustar.' + p_of_S_gt_3;
    nup = [nu_root+D(k)*log(ps(1)) ; nu_p(:) ; nup_gt_3(3:end)]; %including nu_root to include the case when no impulse is present in cluster    %
    p_nup = exp(nup-max(nup))/sum(exp(nup-max(nup)));
    xls_k_mmse(:,k) = [zeros(m,1) xls{k}{:} xxls{3:end}]*p_nup;
    
    %MAP
    [~,indx_max_nup] = max(nup);
    p_nup_map = zeros(size(nup));
    p_nup_map(indx_max_nup) = 1;
    xls_k_map(:,k) = [zeros(m,1) xls{k}{:} xxls{3:end}]*p_nup_map;
    
end

xls_mmse_final = sum(xls_k_mmse,2);
xls_map_final = sum(xls_k_map,2);
