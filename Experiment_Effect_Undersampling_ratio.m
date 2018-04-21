% Effect of the Under-Sampling Ratio
%
% This code demonstrates the effect of under-sampling ratio "us" on the
% performance of the algorithm described in "Structure-Based Bayesian
% Sparse Reconstruction, Ahmed A. Quadeer & Tareq Y. Al-Naffouri." and 
% compares it with other sparse reconstruction algorithms.
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% OC_Gaussian_seq version 1.0
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

clear all;close all;clc
N = 800;        % Length of time vector
us = 4:1:15;    % Undersampling ratio N/M
s = 8;          % Sparsity level or number of impulses in time domain
p = s/N;        % Sparsity rate
F = 1/sqrt(N)*exp(-sqrt(-1)*(2*pi)*(0:1:N-1)'*(0:1:N-1)/N); %DFT Matrix
ITER = 5000;    % Number of iterations
SNR_dB = 30;    % SNR in dB
N0 = 1;         % Noice variance
sigma_imp = 10^(SNR_dB/10)*N0;  % Sparse signal variance

%Initialization
nmse_l1_ref = zeros(1,length(us));
nmse_fbmp = zeros(1,length(us));
nmse_ocmg = zeros(1,length(us));
nmse_omp_ref = zeros(1,length(us));

for ss = 1:1:length(us)
    
    error_l1_ref = zeros(1,ITER);
    error_Schniter = zeros(1,ITER);
    error_ocmg = zeros(1,ITER);
    error_omp_ref = zeros(1,ITER);
       
    for iter = 1:ITER
        
        %% Constructing the sparse signal
        
        x = zeros(N,1);
        impulse_place = randi(N,1,s);  %takes s random values out of N
        x(impulse_place) = sqrt(sigma_imp/2)*(randn(size(impulse_place)) + 1i*randn(size(impulse_place)));
               
       
        %% Measured Signal
        
        AA = eye(N);
        M = ceil(N/us(ss));                        %Number of Measurements
        index= 1:M;                                %Index of measurements
        A = AA(index,:);
        Psi = A*F;                                 %Measurement matrix
        n = sqrt(N0/2)*randn(M,1)+1i*randn(M,1);   %Noise vecror
        y = Psi*x + n;                             %Measurement vector      
        
        %% L1 norm minimization using CVX (Given b and A, find x_hat such that A*x_hat = b)
        
        cvx_begin
            cvx_quiet(true)                     % For supressing results displayed in command window
            variable x_hat(N) complex;          % Defining the estimation variable with its length in brackets
            epsilon = sqrt(N0*(M+2*sqrt(2*M)));
        
            minimize(norm(x_hat,1));            % Problem Definition
            subject to
            norm(y-Psi*x_hat,2) <= epsilon;
        cvx_end
        
        % Support Refinement
        J = BLC_Gaussian(x_hat,y,p,Psi',N0,sigma_imp);
        % Amplitude Refinement
        x_hat_ref = Refinement_MMSE(J,y,Psi',N0,sigma_imp);
        
        error_l1_ref(iter) = norm(x-x_hat_ref)^2/norm(x)^2;
                
        %% Schniter
        
        sig2s = [0; sigma_imp];     % sparse coefficient variances [off;on]
        mus = [0; 0];               % sparse coefficient means [off;on;...;on]
        D = 10;
        stop = 0;
        xmmse_best = fbmpc_fxn_reduced(y, Psi, p, N0, sig2s, mus, D, stop);
        error_Schniter(iter) = norm(x-xmmse_best)^2/norm(x)^2;
        
        %% OC Gaussian
        
        x_ocmg = OC_Gaussian_seq(y,Psi,sigma_imp,N0,p,index);
        error_ocmg(iter) = norm(x-x_ocmg)^2/norm(x)^2;
        
        %% OMP
        
        sup_omp = greed_omp(y,Psi,N);
        % Support Refinement
        J_omp = BLC_Gaussian(sup_omp,y,p,Psi',N0,sigma_imp);
        % Amplitude Refinement
        x_omp_ref = Refinement_MMSE(J_omp,y,Psi',N0,sigma_imp);
        error_omp_ref(iter) = norm(x-x_omp_ref)^2/norm(x)^2;        
        
    end
    
    nmse_l1_ref(ss) = 10*log10(sum(error_l1_ref)/ITER)
    nmse_fbmp(ss) = 10*log10(sum(error_Schniter)/ITER)
    nmse_ocmg(ss) = 10*log10(sum(error_ocmg)/ITER)
    nmse_omp_ref(ss) = 10*log10(sum(error_omp_ref)/ITER)
   
end

%% Plot
figure
plot(us,nmse_fbmp,'r-.o',us,nmse_l1_ref,'g:d',...
    us,nmse_ocmg,'b-s',...
    us,nmse_omp_ref,'k--p','LineWidth',2)
legend('FBMP','CR','OC','OMP')
xlabel('Under-sampling ratio (N/M)')
ylabel('NMSE (dB)')
