function Y = hyperfilter_R(m,t)
% Calculate Needlet Filter polynomial p on [0, 1]
% p(0) = 1, p(1) = 0
% First m+1 derivatives are zero at x = 0 and 
% first m derivaitves are zero at x = 1
% Degree of polynomial is n = 2*m + 2
% p(x) = sum_{k=m+1}^n a_k * (x - 1)^k
%
% Needlet filter h has 0 <= h(s) <= 1 for all s and support [1/2, 2]
% h(s) = p(s-1) for s in [1, 2]
% h(s) = sqrt(1-p(2*s-1)^2) for s in [1/2, 1]
% As h(s)^2 + h(2*s)^2 = 1 for s in [1/2, 1]
%    h(s)*h'(s) + 2*h(2*s)*h'(2*s) = 0 
% h(1/2) = 0 implies p need m+1 deriviatives 0 at 0 to get h in C^m

logic_t_2 = t>1;
Y_1=ones(size(t));

Y_1(logic_t_2) = (Needletfilterpoly(m,t(logic_t_2))).^2;

Y=Y_1;