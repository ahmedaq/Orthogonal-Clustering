% BLC_NonGaussian - BLC = Best Largest Candidate
% Function to refine the support using the a priori information that the 
% signal is non-Gaussian distributed
% Inputs:   e = input signal (Nx1)
%           yp = Observation/Measurement vector (Mx1)
%           pp = sparsity rate (Scalar)
%           Phi_2 = Measurement matrix (NxM)
%           N0 = var. of noise (Scalar)
% Output:   J = Refined support based on x|supp. non-Gaussian
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 18, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012


function J = BLC_NonGaussian(e,yp,pp,Phi_2,N0)
[m,n] = size(Phi_2);
[~,S2] = sort(abs(e).^2);
z_max = 20;                 %Number of combinations to explore
L = zeros(1,z_max);         %Likelihood value
for zz = 0:1:z_max
    J = S2(end-(zz-1):end);
    if  zz == 0
        Sigma = eye(n);
        L(zz+1) = zz*log(pp)+(m-zz)*log(1-pp)-norm(Sigma*yp)^2/N0;
    else
        Psi = Phi_2(J,:)';
        Sigma = eye(n) - Psi/(Psi'*Psi)*Psi';
        L(zz+1) = zz*log(pp)+(m-zz)*log(1-pp)-norm(Sigma*yp)^2/N0;
    end
end

[~,b1] = max(real(L));
J = S2(end-b1+2:end);