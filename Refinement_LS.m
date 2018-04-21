% Refinement_LS - 
% Function to refine the amplitudes (given the support set) using the LS criteria
%
% Inputs:   J = Refined support (Nx1)
%           y = Observation/Measurement vector (Mx1)
%           Phi_2 = Measurement matrix (NxM)
% Output:   e = Signal with refined amplitudes at the supp. locations J 
% 
% Coded by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: Dec. 18, 2012
% Copyright (c) Ahmed Abdul Quadeer, Tareq Y. Al-Naffouri, 2012

function e = Refinement_LS(J,y,Phi_2)
[m,n] = size(Phi_2);
if  isempty(J)
    e = 0;
else
    %ddd = ddd + 1
    e = zeros(m,1);
    Phi_2_J = Phi_2(J,:);
    q = inv(Phi_2_J*Phi_2_J')*Phi_2_J*y;
    e(J) = q;
end