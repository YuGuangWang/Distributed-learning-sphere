function FLY = FilteredSphKer_r(d,filter_m,filterType,n,t)
% FLY = FilteredSphKer_g(d,gamma_1,filterNo,filterType,n,t)
% computes the values of the filtered kernel 
% v_{n,h}^{(d)} = \sum_{\ell=0}^{\infty}Z(\ell,d)P_{\ell}^{(d)}(t)
% of degree n at t with filter of number 'filterNo' and type of
% @filterType. 
% To accelarate the computation, we use this alternative expression for 
% each term: 
% Z(ell,d)P_{ell}^{(d)}(t) = 
%   sum_{ell} a_ell*JacobiRecursiveT2(alpha_d,beta_d,ell,t),
%   where a_ell = (2-1/(1+ell/(d-1)))*2*Prod(2-1./(1+d./(2*j))) for ell>=2.
%
% Inputs:
% d -- dimension of the sphere S^d
% gamma_1 -- truncation constant
% filterNo -- number of the primitive filter
% filterType -- type of the filter
% n -- order of the kernel
% t -- independent variable of the kernel
% Outputs:
% FLY -- values of the filtered kernel at t with same size of t
if n<1&&n>=0
    % by definition, when 0<=n<1, v_{n,h}^d==1
    FLY =  ones(size(t));
else
alpha1=(d-2)/2;
beta1=(d-2)/2;
Lgamma=max(ceil(2*n-1),n);
ell=0:Lgamma;
%a_ell=zeros(length(ell),1);
a_1 = filterType(filter_m,ell(1)/n);
a_2 = filterType(filter_m,ell(2)/n)*(d+1)/d*2;
% compute the first two terms of the filtered kernel Z(d,ell) P_ell^(d)(X)
Y=a_1*ones(size(t))+a_2*JacobiRecursiveT2(alpha1,beta1,1,t);
for i=3:length(ell)
    Coe_ell_tmp=1:(ell(i)-1);   
    Coe_ell = 2*prod(2-1./(1+d./(2*Coe_ell_tmp)));
    % compute the coefficients of the filtered kernel before the Jacobi polynomial
    a_ell = filterType(filter_m,ell(i)/n)*(2-1/(1+ell(i)/(d-1)))*Coe_ell;
    Y = Y+a_ell*JacobiRecursiveT2(alpha1,beta1,ell(i),t);
end
% compute the Dirichlet kernel
FLY=Y;
end