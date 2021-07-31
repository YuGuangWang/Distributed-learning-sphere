function Y = filterpoly_1(m,xp)
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

%format compact

% Coefficient of linear system for a_k k = m+1:n
ac = @(k,m) (-1).^(k-m) .* factorial(k)./factorial(k-m);

% Derivatives specified at x = 0 and x = 1
% m+1 derivatives specified to be 0 at x = 0
% m   deriviatves specified at be 0 at x = 1
% Numerical difficulties for m >= 8
% m = 5;
if m==5
%     ld=['filterpoly_m5' '.mat']
%     load(ld);
ar=[924;
        4752;
       10395;
       12320;
        8316;
        3024;
         462];
% Degree of polynomial n = 2*m+2
n = 2*m + 2;
else

% Degree of polynomial n = 2*m+2
n = 2*m + 2;

% Indicies of coefficients to be found
I = m+1:n;

% Coefficient matrix
A = zeros(m+2, m+2);
for k = m+1:n
    jj = k - m;
    A(jj,:) = ac(I,jj-1);
end;
% A
% Acond = cond(A)

% RHS
b = zeros(m+2,1);
b(1) = 1;

% Solution to linear system
% fprintf('\nFilter polynomial p(x) = sum_{k=1}^{m+2} a_k * (x - 1)^{m+k}\n')

a = A \ b;
% coeff = a';
% r = A*a - b;
% fprintf('Infinity norm of residual = %.2e\n', norm(r, inf))

% Check for integer solution
ar = round(a);
% ar = a;
% coeff_rounded = ar';
% rr = A*ar - b;
% fprintf('Infinity norm of residual for rounded coefficients = %.2e\n', norm(rr, inf))

% Define polynomial for plotting
% xp = linspace(0, 1, 10001);
end
p = zeros(size(xp));
for k = m+1:n
    kj = k-m;
    p = p + ar(kj)*(xp-1).^k;
end;
Y = p;