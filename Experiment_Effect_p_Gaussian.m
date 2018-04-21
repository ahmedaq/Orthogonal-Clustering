% Effect of the sparsity rate given sparse signal is Gaussian distributed
%
% This code demonstrates the effect of sparsity rate "p" on the
% performance of the algorithm described in "Structure-Based Bayesian
% Sparse Reconstruction, Ahmed A. Quadeer & Tareq Y. Al-Naffouri." and 
% compares it with other sparse reconstruction algorithms for the case when
% the sparse signal is Gaussian distributed.
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

clear all;close all;clc
N = 1000;           % Length of sparse signal
M = N/4;            % Number of Measurements
s = 1:2:20;         % Sparsity level
F = 1/sqrt(N)*exp(-sqrt(-1)*(2*pi)*(0:1:N-1)'*(0:1:N-1)/N); %DFT Matrix
ITER = 5000;        % Number of iterations
p = s./N;           % Sparsity rate
N0 = 1;             % Noice variance
%Initialization
nmse_l1_ref = zeros(1,length(s));
nmse_fbmp = zeros(1,length(s));
nmse_ocmg = zeros(1,length(s));
nmse_omp_ref = zeros(1,length(s));
t_l1_ref = zeros(1,length(s));
t_fbmp = zeros(1,length(s));
t_ocmg = zeros(1,length(s));
t_omp_ref = zeros(1,length(s));
for ss = 1:1:length(s)
    
    %Initialization
    error_l1_ref = zeros(1,ITER);
    error_fbmp = zeros(1,ITER);
    error_ocmg = zeros(1,ITER);
    error_omp_ref = zeros(1,ITER);
    
    time_l1_ref = zeros(1,ITER);
    time_fbmp = zeros(1,ITER);
    time_ocmg = zeros(1,ITER);
    time_omp_ref = zeros(1,ITER);
    
    for iter = 1:ITER
        
        %% Constructing the sparse signal
        
        x = zeros(N,1);
        impulse_place = randi(N,1,s(ss));   % Takes s random values out of N
        sigma_imp = 1000;                   % Sparse signal variance
        x(impulse_place) = sqrt(sigma_imp/2)*(randn(size(impulse_place)) + 1i*randn(size(impulse_place)));       
                                
        %% Measured Signal
        
        AA = eye(N);
        index= 1:M;
        A = AA(index,:);
        Psi = A*F;                                 % Measurement matrix
        n = sqrt(N0/2)*randn(M,1)+1i*randn(M,1);   % Noise vecror
        y = Psi*x + n;                             % Measurement vector
        
        %% L1 norm minimization using CVX (Given b and A, find x_hat such that A*x_hat = b)
        
        tic
        cvx_begin
        cvx_quiet(true)                    % For supressing results displayed in command window
        variable x_hat(N) complex;         % Defining the estimation variable with its length in brackets
        epsilon = sqrt(N0*(M+2*sqrt(2*M)));
        
        minimize(norm(x_hat,1));           % Problem Definition
        subject to
        norm(y-Psi*x_hat,2) <= epsilon;
        cvx_end
        
        J = BLC_Gaussian(x_hat,y,p(ss),Psi',N0,sigma_imp);      % Support Refinement (x|Supp. Gaussian)
        
        x_hat_ref = Refinement_MMSE(J,y,Psi',N0,sigma_imp); % Amplitude Refinement (MMSE)
        
        time_l1_ref(iter) = toc;
        error_l1_ref(iter) = norm(x-x_hat_ref)^2/norm(x)^2;
        
        %% fbmp
        
        sig2s = [0; sigma_imp];     % Sparse coefficient variances [off;on]
        mus = [0; 0];               % sparse coefficient means [off;on;...;on]
        D = 10;                     % Greedy branches
        stop = 0;                   % Stop criteria
        
        tic
        x_fbmp = fbmpc_fxn_reduced(y, Psi, p(ss), N0, sig2s, mus, D, stop);
        
        time_fbmp(iter) = toc;
        
        error_fbmp(iter) = norm(x-x_fbmp)^2/norm(x)^2;
        
        %% OC
        
        tic
        x_ocmg = OC_Gaussian_seq(y,Psi,sigma_imp,N0,p(ss),index);
        
        time_ocmg(iter) = toc;
        
        error_ocmg(iter) = norm(x-x_ocmg)^2/norm(x)^2;
                
        %% OMP
        
        tic
        sup_omp = greed_omp(y,Psi,N);
        
        J_omp = BLC_Gaussian(sup_omp,y,p(ss),Psi',N0,sigma_imp); % Support Refinement
        
        x_omp_ref = Refinement_MMSE(J_omp,y,Psi',N0,sigma_imp); % Amplitude Refinement
        
        time_omp_ref(iter) = toc;
        
        error_omp_ref(iter) = norm(x-x_omp_ref)^2/norm(x)^2;
                
    end
    
    %Update
    nmse_l1_ref(ss) = 10*log10(sum(error_l1_ref)/ITER)
    nmse_fbmp(ss) = 10*log10(sum(error_fbmp)/ITER)
    nmse_ocmg(ss) = 10*log10(sum(error_ocmg)/ITER)
    nmse_omp_ref(ss) = 10*log10(sum(error_omp_ref)/ITER)
        
    t_l1_ref(ss) = sum(time_l1_ref)/ITER
    t_fbmp(ss) = sum(time_fbmp)/ITER
    t_ocmg(ss) = sum(time_ocmg)/ITER
    t_omp_ref(ss) = sum(time_omp_ref)/ITER
    
end

%% Plots

figure
plot(p,nmse_l1_ref,'g:d',p,nmse_fbmp,'r-.o',p,nmse_ocmg,'b-s',...
    p,nmse_omp_ref,'k--p','LineWidth',2)
legend('CR','FBMP','OC','OMP')
axis([0 0.02 -25 0])
xlabel('\bf \it p')
ylabel('\bf NMSE (dB)')
grid on

figure
semilogy(p,t_l1_ref,'g:d',p,t_fbmp,'r-.o',p,t_ocmg,'b-s',...
    p,t_omp_ref,'k--p','LineWidth',2)
legend('CR','FBMP','OC','OMP')
xlabel('\bf \it p')
ylabel('\bf Mean run-time')
grid on