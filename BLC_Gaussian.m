% BLC_Gaussian - BLC = Best Largest Candidate
% Function to refine the support using the a priori information that the 
% signal is Gaussian distributed
% Inputs:   e = input signal (Nx1)
%           yp = Observation/Measurement vector (Mx1)
%           pp = sparsity rate (Scalar)
%           Phi_2 = Measurement matrix (NxM)
%           N0 = var. of noise (Scalar)
%           sigma_imp = var. of non-zero values (Scalar)
% Output:   J = Refined support based on x|supp. Gaussian
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 18, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

function J = BLC_Gaussian(e,yp,pp,Phi_2,N0,sigma_imp)
[m,n] = size(Phi_2);
[~,S2] = sort(abs(e).^2);
z_max = 20;                 %No. of combinations to explore
L = zeros(1,z_max);         %Likelihood value
for zz = 0:1:z_max
    J = S2(end-(zz-1):end);
    if  zz == 0
        Sigma = eye(n);
        L(zz+1) = zz*log(pp)+(m-zz)*log(1-pp)-log(det(Sigma))-yp'*inv(Sigma)*yp/N0;
    else
        Psi = Phi_2(J,:)';
        Sigma = eye(n)+(sigma_imp/N0)*Psi*Psi';
        L(zz+1) = zz*log(pp)+(m-zz)*log(1-pp)-log(det(Sigma))-yp'*inv(Sigma)*yp/N0;
    end
end

[~,b1] = max(real(L));
J = S2(end-b1+2:end);