% Effect of the sparsity rate given sparse signal is non-Gaussian distributed
%
% This code demonstrates the effect of sparsity rate "p" on the
% performance of the algorithm described in "Structure-Based Bayesian
% Sparse Reconstruction, Ahmed A. Quadeer & Tareq Y. Al-Naffouri." and 
% compares it with other sparse reconstruction algorithms for the case when
% the sparse signal is non-Gaussian distributed.
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

clear all;close all;clc
N = 1000;                   % Length of sparse signal
M = N/4;                    % Number of Measurements
F = 1/sqrt(N)*exp(-sqrt(-1)*(2*pi)*(0:1:N-1)'*(0:1:N-1)/N); %DFT Matrix
ITER = 5000;                % Number of iterations
N0 = 1;                     % Noice variance
s = 1:2:20;                 % Sparsity level
SNR_dB = 30;                % SNR in dB
sigma_imp = 10^(SNR_dB/10); % Sparse signa variance

%Initialization

nmse_l1_ref = zeros(1,length(s));
nmse_fbmp = zeros(1,length(s));
nmse_ocmng = zeros(1,length(s));
nmse_omp_ref = zeros(1,length(s));

ntime_l1_ref = zeros(1,length(s));
ntime_fbmp = zeros(1,length(s));
ntime_ocmng = zeros(1,length(s));
ntime_omp_ref = zeros(1,length(s));

for ss = 1:length(s)
    
    p = s(ss)/N;        % Sparsity rate
    
    %Initialization
    error_l1_ref = zeros(1,ITER);
    error_fbmp = zeros(1,ITER);
    error_ocmng = zeros(1,ITER);
    error_omp_ref = zeros(1,ITER);    
    
    time_l1_ref = zeros(1,ITER);
    time_fbmp = zeros(1,ITER);
    time_ocmng = zeros(1,ITER);
    time_omp_ref = zeros(1,ITER);
    
    for iter = 1:ITER
        
        %% Constructing the sparse signal
        
        x = zeros(N,1);
        impulse_place = randi(N,1,s(ss));  %Takes s random values out of N
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
        
        tic
        cvx_begin
        cvx_quiet(true)                     % For supressing results displayed in command window
        variable x_hat(N) complex;          % Defining the estimation variable with its length in brackets
        epsilon = sqrt(N0*(M+2*sqrt(2*M)));
        
        minimize(norm(x_hat,1));            % Problem Definition
        subject to
        norm(y-Psi*x_hat,2) <= epsilon;
        cvx_end
        
        % Support Refinement
        J = BLC_NonGaussian(x_hat,y,p,Psi',N0);
        
        % Amplitude Refinement
        x_hat_ref = Refinement_LS(J,y,Psi');
        
        time_l1_ref(iter) = toc;
        error_l1_ref(iter) = norm(x-x_hat_ref)^2/norm(x)^2;
        
        
        %% FBMP
        
        sig2s = [0; 1e-10];     % sparse coefficient variances [off;on]
        mus = [5; 20];          % sparse coefficient means [off;on;...;on]
        D = 10;                 % no. of greedy searches
        stop = 0;
        E = 5;                  % max. no of refinement stages
        
        tic
        x_fbmp = ...
            fbmpc_gem_refine_fxn(y, Psi, p, N0, sig2s, mus, D, stop, E);
        time_fbmp(iter) = toc;
        error_fbmp(iter) = norm(x-x_fbmp)^2/norm(x)^2;       
        
        
        %% OCMNG
               
        tic
        [xls_mmse_final_seq_1,xls_map_final_seq_1] = ...
            OC_nonGaussian_seq(y,Psi,N0,p,index);
        
        time_ocmng(iter) = toc;
        error_ocmng(iter) = norm(x-xls_map_final_seq_1)^2/norm(x)^2;      
        
        %% OMP
        
        tic
        [sup_omp, err_norm, iter_time] = greed_omp(y,Psi,N);
        
        % Support Refinement
        J_omp = BLC_NonGaussian(sup_omp,y,p,Psi',N0);
        % Amplitude Refinement
        x_omp = Refinement_LS(J_omp,y,Psi');
        
        time_omp_ref(iter) = toc;
        error_omp_ref(iter) = norm(x-x_omp)^2/norm(x)^2;
        
    end
    
    %Update
    nmse_l1_ref(ss) = 10*log10(sum(error_l1_ref)/ITER)
    nmse_fbmp(ss) = 10*log10(sum(error_fbmp)/ITER)
    nmse_ocmng(ss) = 10*log10(sum(error_ocmng)/ITER)    
    nmse_omp_ref(ss) = 10*log10(sum(error_omp_ref)/ITER)
    
    ntime_l1_ref(ss) = sum(time_l1_ref)/ITER
    ntime_fbmp(ss) = sum(time_fbmp)/ITER
    ntime_ocmng(ss) = sum(time_ocmng)/ITER   
    ntime_omp_ref(ss) = sum(time_omp_ref)/ITER
end

%% Plots

pp = s./N;

figure;
plot(pp,nmse_fbmp,'r-.o',pp,nmse_l1_ref,'g:d',...
    pp,nmse_ocmng,'b-s',pp,nmse_omp_ref,'k--s','LineWidth',2)
legend('FBMP','CR','OC','OMP')
xlabel('\bf \it p')
ylabel('\bf NMSE (dB)')
grid on

figure
semilogy(pp,ntime_fbmp,'r-.o',pp,ntime_l1_ref,'g:d',...
    pp,ntime_ocmng,'b-s',pp,ntime_omp_ref,'k--s','LineWidth',2)
legend('FBMP','CR','OC','OMP')
xlabel('\bf \it p')
ylabel('\bf Mean run-time')
grid on
