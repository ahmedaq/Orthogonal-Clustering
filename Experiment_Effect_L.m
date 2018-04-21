% Effect of the cluster length L
%
% This code demonstrates the effect of cluster length "L" on the
% performance of the algorithm described in "Structure-Based Bayesian
% Sparse Reconstruction", Ahmed A. Quadeer & Tareq Y. Al-Naffouri.
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 12, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

clear all;close all;clc

N = 800;        %Length of time vector
M = N/4;        %Number of Measurements
F = 1/sqrt(N)*exp(-sqrt(-1)*(2*pi)*(0:1:N-1)'*(0:1:N-1)/N); %DFT Matrix
ITER = 5000;    %Number of iterations 
s = 1:1:8;      %Number of non-zero values 
p = s./N;       %Sparsity rate
SNR_dB = 30;    %SNR in dB    
N0 = 1;         %Noice variance
sigma_imp = 10^(SNR_dB/10)*N0;  %Variance of sparse signal

%Initialization
nmse_ocmg_L4 = zeros(1,length(s));
nmse_ocmg_L8 = zeros(1,length(s));
nmse_ocmg_L16 = zeros(1,length(s));
nmse_ocmg_L32 = zeros(1,length(s));
nmse_ocmg = zeros(1,length(s));

t_ocmg_L4 = zeros(1,length(s));
t_ocmg_L8 = zeros(1,length(s));
t_ocmg_L16 = zeros(1,length(s));
t_ocmg_L32 = zeros(1,length(s));
t_ocmg = zeros(1,length(s));

for ss = 1:1:length(s)
    
    %Intialization
    error_ocmg_L4 = zeros(1,ITER);
    error_ocmg_L8 = zeros(1,ITER);
    error_ocmg_L16 = zeros(1,ITER);
    error_ocmg_L32 = zeros(1,ITER);    
    error_ocmg = zeros(1,ITER);

    time_ocmg_L4 = zeros(1,ITER);
    time_ocmg_L8 = zeros(1,ITER);
    time_ocmg_L16 = zeros(1,ITER);
    time_ocmg_L32 = zeros(1,ITER);    
    time_ocmg = zeros(1,ITER);
    
    for iter = 1:ITER
        
        %% Constructing the sparse signal
        
        x = zeros(N,1);
        impulse_place = randi(N,1,s(ss));       %Takes s random values out of N
        x(impulse_place) = sqrt(sigma_imp/2)*(randn(size(impulse_place)) + 1i*randn(size(impulse_place)));                
        
        %% Measured Signal
        
        AA = eye(N);
        index= 1:M;
        A = AA(index,:);
        Psi = A*F;                                 %Measurement matrix
        n = sqrt(N0/2)*randn(M,1)+1i*randn(M,1);   %Noise vecror
        y = Psi*x + n;                             %Measurement vector       
              
        %% OC Gaussian Fixed Cluster
        
        L = 4;
        tic
        x_ocmg_L4 = OC_Gaussian_fixed(y,Psi,sigma_imp,N0,p(ss),index,L);
        time_ocmg_L4(iter) = toc;
        error_ocmg_L4(iter) = norm(x-x_ocmg_L4)^2/norm(x)^2;
                
        %        
        L = 8;
        tic
        x_ocmg_L8 = OC_Gaussian_fixed(y,Psi,sigma_imp,N0,p(ss),index,L);
        time_ocmg_L8(iter) = toc;
        error_ocmg_L8(iter) = norm(x-x_ocmg_L8)^2/norm(x)^2;
        
        %        
        L = 16;
        tic
        x_ocmg_L16 = OC_Gaussian_fixed(y,Psi,sigma_imp,N0,p(ss),index,L);
        time_ocmg_L16(iter) = toc;
        error_ocmg_L16(iter) = norm(x-x_ocmg_L16)^2/norm(x)^2;
        
        %        
        L = 32;
        tic
        x_ocmg_L32 = OC_Gaussian_fixed(y,Psi,sigma_imp,N0,p(ss),index,L);
        time_ocmg_L32(iter) = toc;
        error_ocmg_L32(iter) = norm(x-x_ocmg_L32)^2/norm(x)^2;
        
        %% OC Gaussian
        
        tic
        x_ocmg = OC_Gaussian_seq(y,Psi,sigma_imp,N0,p(ss),index);
        error_ocmg(iter) = norm(x-x_ocmg)^2/norm(x)^2;
        time_ocmg(iter) = toc;      
        
    end
    
    nmse_ocmg_L4(ss) = 10*log10(sum(error_ocmg_L4)/ITER)
    nmse_ocmg_L8(ss) = 10*log10(sum(error_ocmg_L8)/ITER)
    nmse_ocmg_L16(ss) = 10*log10(sum(error_ocmg_L16)/ITER)
    nmse_ocmg_L32(ss) = 10*log10(sum(error_ocmg_L32)/ITER)
    nmse_ocmg(ss) = 10*log10(sum(error_ocmg)/ITER)
    
    t_ocmg_L4(ss) = sum(time_ocmg_L4)/ITER
    t_ocmg_L8(ss) = sum(time_ocmg_L8)/ITER
    t_ocmg_L16(ss) = sum(time_ocmg_L16)/ITER
    t_ocmg_L32(ss) = sum(time_ocmg_L32)/ITER
    t_ocmg(ss) = sum(time_ocmg)/ITER
    
end

figure
plot(p,nmse_ocmg_L4,'b:<',p,nmse_ocmg_L8,'b:>',...
    p,nmse_ocmg_L16,'b:^',p,nmse_ocmg_L32,'b:v',...
    p,nmse_ocmg,'b-s','LineWidth',2)
legend('OC, L=4','OC, L=8','OC, L=16','OC, L=32','OC')
xlabel('\bf \it p')
ylabel('\bf NMSE (dB)')
grid on

figure
semilogy(p,t_ocmg_L4,'b:<',p,t_ocmg_L8,'b:>',...
    p,t_ocmg_L16,'b:^',p,t_ocmg_L32,'b:v',...
    p,t_ocmg,'b-s','LineWidth',2)
legend('OC, L=4','OC, L=8','OC, L=16','OC, L=32','OC')
xlabel('\bf \it p')
ylabel('\bf Mean run-time')
grid on
