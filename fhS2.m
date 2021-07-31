function y = fhS2(n,t,m)
% computes the filtered kernel for filtered hyperinterpolation
% INPUTS:
%  n - degree of filtered kernel
%  t - matrix for variable of the filtered kernel, size(t) = [M,N]
%  m - smoothness of filter, h\in W^m
% OUTPUTS:
%  y - values of filtered kernel, size(y) = size(t)

if nargin < 3
    m = 5;
end

% filter
fih = @(t) hyperfilter_R(m,t);

% filtered kernel at t with filtere fih
ykn = legendre(0,t)/(4*pi);
for l = 1:2*n-1
    % legendre polynomial
    ylg = legendre(l,t);
    ylg1 = zeros(size(t));
    ylg1(:,:) = ylg(1,:,:);
    ykn = ykn + (2*l+1)/(4*pi)*fih(l/n)*ylg1;
end
y = ykn;
end
    
    