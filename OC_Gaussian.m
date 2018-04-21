% OC_Gaussian - Function for executing the complex-valued MMSE estimate of
% a Gaussian distributed sparse signal using orthoganal clustering approach.
%
% This function implements the algorithm described in "Structure-Based 
% Bayesian Sparse Reconstruction", Ahmed A. Quadeer & Tareq Y. Al-Naffouri.
% OC_Gaussian will find a MMSE estimate of sparse solutions "x"
% to the underdetermined system of linear equations
%                           yp = Psi*x + n,
% where n is complex Gaussian, while x is Gaussian distributed.
%
% SYNTAX: xmmse_final = OC_Gaussian(yp, Psi, sigma_imp, N0, pp, index)
%
% Outputs:  xmmse_final = The MMSE estimate of the sparse vector%           
%
% Inputs:   yp = n x 1 observation vector
%           Psi = n x m measurement/sensing matrix
%           sigma_imp = Non-zero elements' variance in the sparse vector
%           N0 = Noise variance
%           pp = Probability of non-zero elements occurence
%           index = n-length vector representing indices of continuous
%           sensing carriers
%
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

function xmmse_final = OC_Gaussian(yp,Psi,sigma_imp,N0,pp,index)

%% Initialization
sig2s = [0; sigma_imp];     %Signal variance
sig2w = N0;                 %Noise variance

%% Step 1: Determine dominant positions using correlation

Phi_mtrx = Psi';
[n,m] = size(Phi_mtrx);     %Size of Phi

yp_corr = Phi_mtrx'*yp;     %Performing correlation

% Pz = 1e-2;                %Probability of noise to be greater than thresh
thresh = 3;                 %Calculated from pdf of chisquare distribution

indx_gt_thresh = find(abs(yp_corr)>thresh);  %Indices greater than threshold

%% Step 2: Form semi-orthogonal clusters using chi-square threshold

if isempty(indx_gt_thresh) %If no index greater than thereshold
    
    xmmse_final=zeros(m,1);
    
else
    [J_cluster,cluster_indices] = clustering(indx_gt_thresh,m);
    
%% Step 3: Find the dominant supports and their likelihoods (Main algorithm)
    
    %Initialization
    D = cellfun('length',J_cluster).';    %Finding length of each cluster
    P = length(D);                          %Number of clusters
    P_c = 2;                                %Initial P for cluster
    AA = Psi';
    A = AA(:,cluster_indices(1:D(1)));      %Measurement matrix corresponding to first cluster
        
    T = cell(P,1);              %Indices of active taps
    nu = cell(P,1);             %Metrics
    xmmse = cell(P,1);          %Initial MMSE estimate of x
    Omega = cell(D(1),P_c+1);     %
    Xi = cell(D(1),P_c+1);  %
    xmmse_p = zeros(m,P);       %MMSE estimate of each cluster p
    
    for p = 1:P                 %Loop for each cluster p
        
        ppp = pp/m*D(p);
        ps = [1 - ppp; ppp];
        
        T{p} = cell(D(p),P_c);          %Indices of active taps in cluster p
        nu{p} = -inf*ones(D(p),P_c);	%Metrics in cluster p
        xmmse{p} = cell(D(p),P_c);      %Initial MMSE estimate of x in cluster p
        
        if p == 1                       %If first cluster
            Omega_root_1 = A/sig2w;
            Xi_root_1 = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A).*Omega_root_1)).^(-1));
            Omega_root = Omega_root_1;
            Xi_root = Xi_root_1;
            z = yp;
        else                            %If subsequent clusters
            Omega_root = Omega_root_1(:,1:(D(p)));                  %Selecting according to length of cluster
            Xi_root = Xi_root_1(1:(D(p)));              %Selecting according to length of cluster
            dd = mod(J_cluster{p}(1)-J_cluster{1}(1),m);        %Finding the difference delta between cluster p and cluster 1
            wd = (exp(-sqrt(-1)*(2*pi)/m*(index(1:n)-1)*dd)).'; %Calculating the modulation vector
            z = yp.*conj(wd);                                   %Modulation
        end
        
        nu_root = -norm(yp)^2/sig2w + D(p)*log(ps(1));  %Root node
        nuxt_root = zeros(1,D(p));
        
        nuxt_root(1:D(p)) = nu_root + log(Xi_root/sig2s(2)) + ...
            Xi_root.*abs(z'*Omega_root).^2 + log(ps(2)/ps(1));
        
        p_c = 1;
        
        for d = 1:D(p)  %Within a cluster
            
            nstar = d;
            T{p}{d,p_c} = J_cluster{p}(nstar);
            nu{p}(d,p_c) = nuxt_root(nstar);
            if p == 1 %Only need to update Omega and Xi for first cluster. Remaining clusters use the same values
                Omega{d,p_c} = Omega_root; %Omega{1,:} will all be equal to Omega_root
                Xi{d,p_c} = Xi_root;
                Omega{d,p_c+1} = Omega{d,p_c} - ...
                    Omega{d,p_c}(:,nstar)*Xi{d,p_c}(nstar)*( Omega{d,p_c}(:,nstar)'*A );
                Xi{d,p_c+1} = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A).*Omega{d,p_c+1})).^(-1));
            end
            xmmse{p}{d,p_c} = zeros(m,1);
            xmmse{p}{d,p_c}(T{p}{d,p_c}) = sig2s(2)*Omega{d,p_c+1}(:,nstar)'*z;
            
            nuxt = zeros(1,D(p));
            nuxt(1:D(p)) = nu{p}(d,p_c) + log(Xi{d,p_c+1}(1:D(p))/sig2s(2)) ...
                + Xi{d,p_c+1}(1:D(p)).*abs(z'*Omega{d,p_c+1}(:,1:D(p))).^2 ...
                + log(ps(2)/ps(1));
            % can't activate an already activated coefficient!
            nuxt(nstar) = -inf*ones(size(T{p}{d,p_c}));
            
            [nustar,nqstar] = max(nuxt);                        % find best extension
            while sum(abs(nustar-nu{p}(1:d-1,p_c+1)) < 1e-8)    % if same as explored node...
                nuxt(nqstar) = -inf;                            % ... mark extension as redundant
                [nustar, nqstar] = max(nuxt);                   % ... and find next best extension
            end
            nstar2 = mod(nqstar - 1, m) + 1;                    % coef index of best extension
            nu{p}(d,p_c+1) = nustar;                            % replace worst explored node...
            
            T{p}{d,p_c+1} = [T{p}{d,p_c}, J_cluster{p}(nstar2)];
            
            Omega{d,p_c+2} = Omega{d,p_c+1}(:,1:D(p)) - ...
                Omega{d,p_c+1}(:,nstar2)*Xi{d,p_c+1}(nstar2)*...
                ( Omega{d,p_c+1}(:,nstar2)'*A(:,1:D(p)));
            Xi{d,p_c+2} = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A(:,1:D(p)))...
                .*Omega{d,p_c+2})).^(-1));
            
            xmmse{p}{d,p_c+1} = zeros(m,1);
            xmmse{p}{d,p_c+1}(T{p}{d,p_c+1}) = sig2s(2)*Omega{d,p_c+2}(:,[nstar,nstar2])'*z;
            
        end
        
        %Finding the best supp. of hamming weight 3 from the best supp. of hamming weight 2
        [a b]=max(nu{p}(:,2)); %a is the max likelihood of supp. of size and b is the index
        nstar = find(J_cluster{p}==T{p}{b,2}(1));
        nstar2 = find(J_cluster{p}==T{p}{b,2}(2));
        nuxt3 = a + log(Xi{b,p_c+2}(1:D(p))/sig2s(2)) ...
            + (Xi{b,p_c+2}(1:D(p))).*abs(z'*Omega{b,p_c+2}(:,1:D(p))).^2 ...
            + log(ps(2)/ps(1));
        nuxt3([nstar nstar2]) = -inf*ones(size(T{p}{b,2}));
        [nustar3,nqstar3] = max(nuxt3);
        while ismember(nqstar3,[nstar nstar2])==1       % if same as explored node...
            nuxt3(nqstar3) = -inf;                      % ... mark extension as redundant
            [nustar3, nqstar3] = max(nuxt3);            % ... and find next best extension
        end
        nstar3 = mod(nqstar3 - 1, m) + 1;
        T_3 = [T{p}{b,2}, J_cluster{p}(nstar3)];
        
        Omega3 = Omega{b,p_c+2}(:,1:D(p)) - Omega{b,p_c+2}(:,nstar3)*Xi{b,p_c+2}...
            (nstar3)*( Omega{b,p_c+2}(:,nstar3)'*A(:,1:D(p)));
        Xi3 = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A(:,1:D(p))).*Omega3)).^(-1));
        
        xmmse_3 = zeros(m,1);
        xmmse_3(T_3) = sig2s(2)*Omega3(:,[nstar,nstar2,nstar3])'*z;
        
        % Continuing the greedy approach for hamming weight greater than 3
        % depending on the length of cluster
        Pc = min(n, 1 + ceil(D(p)*pp + erfcinv(1e-1)*sqrt(2*D(p)*pp*(1 - pp))));
        p_cc = 4;
        nnustar = cell(1,Pc);
        TT = cell(1,Pc);
        Omega_gt_3 = cell(1,Pc);
        Xi_gt_3 = cell(1,Pc);
        NSSTAR = cell(1,Pc);
        xmmsee = cell(1,Pc);
        nnustar{p_cc-1} = nustar3;
        TT{p_cc-1}=T_3;
        Omega_gt_3{p_cc-1} = Omega3;
        Xi_gt_3{p_cc-1}=Xi3;
        NSSTAR{p_cc-1}=[nstar nstar2 nstar3];
        xmmsee{p_cc-1} = xmmse_3;
        clear nxt
        
        while p_cc <= Pc
            
            nxt{p_cc} = zeros(1,D(p));
            nxt{p_cc} = nnustar{p_cc-1} + log(Xi_gt_3{p_cc-1}(1:D(p))/sig2s(2)) ...
                + (Xi_gt_3{p_cc-1}(1:D(p))).*abs(z'*Omega_gt_3{p_cc-1}(:,1:D(p))).^2 + log(ps(2)/ps(1));
            nxt{p_cc}(NSSTAR{p_cc-1}) = -inf*ones(1,p_cc-1);
            [nnustar{p_cc},nnqstar{p_cc}] = max(nxt{p_cc});
            nnstar{p_cc} = mod(nnqstar{p_cc} - 1, m) + 1;
            TT{p_cc} = [TT{p_cc-1} J_cluster{p}(nnstar{p_cc})];
            
            Omega_gt_3{p_cc} = Omega_gt_3{p_cc-1}(:,1:D(p)) - Omega_gt_3{p_cc-1}(:,nnstar{p_cc})* Xi_gt_3{p_cc-1}(nnstar{p_cc})*...
                ( Omega_gt_3{p_cc-1}(:,nnstar{p_cc})'*A(:,1:D(p)));
            Xi_gt_3{p_cc} = abs(sig2s(2)*(1 + sig2s(2)*sum(conj(A(:,1:D(p))).*Omega_gt_3{p_cc})).^(-1));
            
            NSSTAR{p_cc} = [NSSTAR{p_cc-1} nnstar{p_cc}];
            xmmsee{p_cc} = zeros(m,1);
            xmmsee{p_cc}(TT{p_cc}) = sig2s(2)*Omega_gt_3{p_cc}(:,NSSTAR{p_cc})'*z;
            p_cc = p_cc+1;
        end
        
%% Step 4: Evaluate the estimate of x
        
        if Pc > 3
            nup = [nu_root ; nu{p}(:); [nnustar{3:end}].'];
            %including nu_root to include the case when no impulse is present in cluster
            p_nup = exp(nup-max(nup))/sum(exp(nup-max(nup)));
            xmmse_p(:,p) = [zeros(m,1) xmmse{p}{:} [xmmsee{3:end}]]*p_nup;
        else
            nup = [nu_root ; nu{p}(:); nustar3];
            %including nu_root to include the case when no impulse is present in cluster
            p_nup = exp(nup-max(nup))/sum(exp(nup-max(nup)));
            xmmse_p(:,p) = [zeros(m,1) xmmse{p}{:} xmmse_3]*p_nup;
        end
        
        
    end
    
    xmmse_final = sum(xmmse_p,2); %Final MMSE estimate of sparse signal
end