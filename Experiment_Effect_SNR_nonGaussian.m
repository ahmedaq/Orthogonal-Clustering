% SNR vs NMSE for DFT matrix and x|S non-Gaussian
%
% This code demonstrates the effect of SNR on the NMSE 
% performance of the algorithm described in "Structure-Based Bayesian
% Sparse Reconstruction, Ahmed A. Quadeer & Tareq Y. Al-Naffouri." (for the
% case when the sparse signal is non-Gaussian distributed and the sensing
% matrix is a partial DFT matrix) and compares it with other sparse 
% reconstruction algorithms.
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012


clear all;close all;clc
N = 800;            % Length of sparse signal
M = N/4;            % Number of Measurements
F = 1/sqrt(N)*exp(-sqrt(-1)*(2*pi)*(0:1:N-1)'*(0:1:N-1)/N); %DFT Matrix
ITER = 5000;        % Number of iterations
N0 = 1;             % Noice variance
s = 4;              % Sparsity level or number of impulses in time domain
p = s/N;            % Sparsity rate
SNR_dB = 10:3:31;   % SNR in dB

%Initialization
nmse_l1_ref = zeros(1,length(SNR_dB));
nmse_fbmp = zeros(1,length(SNR_dB));
nmse_ocmng = zeros(1,length(SNR_dB));
nmse_omp_ref = zeros(1,length(SNR_dB));

for ss = 1:length(SNR_dB)
    

    error_l1_ref = zeros(1,ITER);
    error_fbmp = zeros(1,ITER);
    error_ocmng = zeros(1,ITER);
    error_omp_ref = zeros(1,ITER);  
    
    for iter = 1:ITER
        
        %% Constructing the sparse signal
        
        x = zeros(N,1);
        impulse_place = randi(N,1,s);  %takes s random values out of N
        sigma_imp = 10^(SNR_dB(ss)/10)*N0;
        x(impulse_place) = sqrt(sigma_imp/2)*(rand(size(impulse_place)) + 1i*rand(size(impulse_place))); %Uniform
        
        %% Measured Signal
        
        AA = eye(N);
        index= 1:M;        
        A = AA(index,:);
        Psi = A*F;                                  %Measurement matrix
        Psi = Psi*diag(1./sqrt(diag(Psi'*Psi)));    %Normalization of measurement matrix
        
        n = sqrt(N0/2)*randn(M,1)+1i*randn(M,1);    %Noise vecror
        y = Psi*x + n;                              %Measurement vector
        
        %% L1 norm minimization using CVX (Given b and A, find x_hat such that A*x_hat = b)
        
        cvx_begin
            cvx_quiet(true)             % For supressing results displayed in command window
            variable x_hat(N) complex;          % Defining the estimation variable with its length in brackets
            epsilon = sqrt(N0*(M+2*sqrt(2*M)));
        
            minimize(norm(x_hat,1));    % Problem Definition
            subject to
            norm(y-Psi*x_hat,2) <= epsilon;
        cvx_end
        
        % Support Refinement
        J = BLC_NonGaussian(x_hat,y,p,Psi',N0);
        
        % Amplitude Refinement
        x_hat_ref = Refinement_LS(J,y,Psi');
        
        error_l1_ref(iter) = norm(x-x_hat_ref)^2/norm(x)^2;
               
        %% FBMP
        
        sig2s = [0; 1e-10];     % sparse coefficient variances [off;on]
        mus = [5; 20];          % sparse coefficient means [off;on;...;on]
        D = 10;                 % no. of greedy searches
        stop = 0;
        E = 5;                  % max no of refinement stages
        
        x_fbmp = ...
            fbmpc_gem_refine_fxn(y, Psi, p, N0, sig2s, mus, D, stop, E);
        error_fbmp(iter) = norm(x-x_fbmp)^2/norm(x)^2;
        
                
        %% OCMG Red. complexity fixed cluster
        
        [xls_mmse_final_seq_1,xls_map_final_seq_1] = ...
            OC_nonGaussian_seq(y,Psi,N0,p,index);
        
        error_ocmng(iter) = norm(x-xls_map_final_seq_1)^2/norm(x)^2;
        
        %% OMP
        
        [sup_omp, err_norm, iter_time] = greed_omp(y,Psi,N);
                
        % Support Refinement
        J_omp = BLC_NonGaussian(sup_omp,y,p,Psi',N0);
        % Amplitude Refinement
        x_omp = Refinement_LS(J_omp,y,Psi');        
        
        error_omp_ref(iter) = norm(x-x_omp)^2/norm(x)^2;
        
    end
    
    %Update
    nmse_l1_ref(ss) = 10*log10(sum(error_l1_ref)/ITER)
    nmse_fbmp(ss) = 10*log10(sum(error_fbmp)/ITER)
    nmse_ocmng(ss) = 10*log10(sum(error_ocmng)/ITER)
    nmse_omp_ref(ss) = 10*log10(sum(error_omp_ref)/ITER)
    
end


figure
plot(SNR_dB,nmse_l1_ref,'g:d',SNR_dB,nmse_omp_ref,'k--p',SNR_dB,nmse_fbmp,'r--o',...
    SNR_dB,nmse_ocmng,'b-s','LineWidth',2)
legend('CR','OMP','FBMP','OC') 
xlabel('\bf SNR (dB)')
ylabel('\bf NMSE (dB)')
axis([10 31 -22 10])
grid on

