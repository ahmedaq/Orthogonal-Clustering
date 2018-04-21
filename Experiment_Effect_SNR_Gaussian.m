% SNR vs NMSE for DFT matrix and x|S Gaussian
%
% This code demonstrates the effect of SNR on the NMSE 
% performance of the algorithm described in "Structure-Based Bayesian
% Sparse Reconstruction, Ahmed A. Quadeer & Tareq Y. Al-Naffouri." (for the
% case when the sparse signal is Gaussian distributed and the sensing
% matrix is a partial DFT matrix) and compares it with other sparse 
% reconstruction algorithms.
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012


clear all;close all;clc

N = 800;            %Length of sparse vector
M = N/4;            %Number of Measurements
s = 4;              %Number of non-zero values
p = s/N;            %Sparsity rate
F = 1/sqrt(N)*exp(-sqrt(-1)*(2*pi)*(0:1:N-1)'*(0:1:N-1)/N); %DFT Matrix
ITER = 5000;        %Number of iterations
N0 = 1;             %Noice variance
SNR_dB = 10:3:31;   %SNR range in dB

%Initialization
nmse_l1_ref = zeros(1,length(SNR_dB));
nmse_FBMP = zeros(1,length(SNR_dB));
nmse_FBMP_MAP = zeros(1,length(SNR_dB));
nmse_ocmg = zeros(1,length(SNR_dB));
nmse_omp_ref = zeros(1,length(SNR_dB));

for ss = 1:length(SNR_dB)           %Loop of SNR
    error_l1_ref = zeros(1,ITER);   %Initialization
    error_FBMP = zeros(1,ITER);
    error_FBMP_MAP = zeros(1,ITER);
    error_ocmg = zeros(1,ITER);
    error_omp_ref = zeros(1,ITER);
    
    for iter = 1:ITER
        
        %% Constructing the sparse signal
        
        x = zeros(N,1);
        impulse_place = randi(N,1,s);       %Takes s random values out of N
        sigma_imp = 10^(SNR_dB(ss)/10)*N0;  %Variance of sparse signal
        x(impulse_place) = sqrt(sigma_imp/2)*(randn(size(impulse_place)) + 1i*randn(size(impulse_place)));       
       
                
        %% Measured Signal
        
        AA = eye(N);
        index= 1:M;
        A = AA(index,:);
        Psi = A*F;                                 %Measurement matrix
        n = sqrt(N0/2)*randn(M,1)+1i*randn(M,1);   %Noise vecror
        y = Psi*x + n;                             %Measurement vector
        
        %% L1 norm minimization using CVX (Given b and A, find x_hat such that A*x_hat = b)
        
        cvx_begin
        cvx_quiet(true)                    %For supressing results displayed in command window
        variable x_hat(N) complex;         %Defining the estimation variable with its length in brackets
        epsilon = sqrt(N0*(M+2*sqrt(2*M)));
        
        minimize(norm(x_hat,1));           %Problem Definition
        subject to
        norm(y-Psi*x_hat,2) <= epsilon;
        cvx_end
        
        J = BLC_Gaussian(x_hat,y,p,Psi',N0,sigma_imp);      %Support Refinement (x|Supp. Gaussian)
        
        x_hat_ref = Refinement_MMSE(J,y,Psi',N0,sigma_imp); %Amplitude Refinement (MMSE)
                
        error_l1_ref(iter) = norm(x-x_hat_ref)^2/norm(x)^2;
        
        %% FBMP
        
        sig2s = [0; sigma_imp];     %Sparse coefficient variances [off;on]
        mus = [0; 0];               %Sparse coefficient means [off;on;...;on]
        D = 10;                     %Greedy branches
        stop = 0;                   %Stop criteria
                       
        
        x_fbmp = fbmpc_fxn_reduced(y, Psi, p, N0, sig2s, mus, D, stop);
        
        error_FBMP(iter) = norm(x-x_fbmp)^2/norm(x)^2;              
                
        %% OC
        
        x_ocmg = OC_Gaussian_seq(y,Psi',sigma_imp,N0,p,index);
        
        error_ocmg(iter) = norm(x-x_ocmg)^2/norm(x)^2;
        
        
        %% OMP
        
        sup_omp = greed_omp(y,Psi,N);
        
        J_omp = BLC_Gaussian(sup_omp,y,p,Psi',N0,sigma_imp);    % Support Refinement (x|Supp. Gaussian)
        
        x_omp = Refinement_MMSE(J_omp,y,Psi',N0,sigma_imp);     % Amplitude Refinement (MMSE)
        
        error_omp_ref(iter) = norm(x-x_omp)^2/norm(x)^2;        
        
    end
    
    %Update
    nmse_l1_ref(ss) = 10*log10(sum(error_l1_ref)/ITER)
    nmse_FBMP(ss) = 10*log10(sum(error_FBMP)/ITER)
    nmse_ocmg(ss) = 10*log10(sum(error_ocmg)/ITER)
    nmse_omp_ref(ss) = 10*log10(sum(error_omp_ref)/ITER)  
    
end

%% Plot

figure
plot(SNR_dB,nmse_l1_ref,'g:d',SNR_dB,nmse_omp_ref,'k--p',SNR_dB,nmse_FBMP,'r--o',...
    SNR_dB,nmse_ocmg,'b-s','LineWidth',2)
legend('CR','OMP','FBMP','OC') 
xlabel('\bf SNR (dB)')
ylabel('\bf NMSE (dB)')
axis([10 31 -22 10])
grid on


% figure
% plot(SNR_dB,nmse_omp_ref,'k--p',SNR_dB,nmse_FBMP,'r--o',...
%     SNR_dB,nmse_ocmg,'b-s','LineWidth',2)
% legend('OMP [17]','FBMP [25]','OC') 
% xlabel('\bf SNR (dB)')
% ylabel('\bf NMSE (dB)')
% axis([10 31 -22 10])
% grid on
