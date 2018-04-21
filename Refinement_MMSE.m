% Refinement_MMSE - 
% Function to refine the amplitudes (given the support set) using the MMSE criteria
%
% Inputs:   J = Refined support (Nx1)
%           y = Observation/Measurement vector (Mx1)
%           Phi_2 = Measurement matrix (NxM)
%           N0 = var. of noise (Scalar)
%           sigma_imp = var. of non-zero values (Scalar)
% Output:   e = Signal with refined amplitudes at the supp. locations J 
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 18, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

function e = Refinement_MMSE(J,y,Phi_2,N0,sigma_imp)

[m,n] = size(Phi_2);
if  isempty(J)
    e = 0;
else
    e = zeros(m,1);
    Phi_2_J = Phi_2(J,:);
    q = sigma_imp*Phi_2_J*((sigma_imp*(Phi_2_J)'*Phi_2_J+N0*eye(n))\y);
    e(J) = q;
end
