function y = fh1(n,t,m)
% computes the filtered kernel for filtered hyperinterpolation
% INPUTS:
%  n - degree of filtered kernel
%  t - matrix for variable of the filtered kernel, size(t) = [M,N]
%  m - smoothness of filter, h\in W^m
% OUTPUTS:
%  y - values of filtered kernel, size(y) = size(t)

% smoothness of filter
if nargin < 3
    m = 5;
end
% filter
fih = @(t) hyperfilter_R(m,t);

% filtered kernel at t with filtere fih
ykn = legen(0,t)/(4*pi);
for l = 1:n
    % legendre polynomial
    ylg = legen(l,t);
    ykn = ykn + fih(l/n)*(2*l+1)/(4*pi)*ylg;
end
y = ykn;
end