function y = hyp1(n,t)
% computes the Fourier kernel on S^2
% INPUTS:
%  n - degree of filtered kernel
%  t - matrix for variable of the filtered kernel, size(t) = [M,N]
% OUTPUTS:
%  y - values of Fourier kernel, size(y) = size(t)

% filtered kernel at t with filtere fih
ykn = legen(0,t)/(4*pi);
for l = 1:n
    % legendre polynomial
    ylg = legen(l,t);
    ykn = ykn + (2*l+1)/(4*pi)*ylg;
end
y = ykn;
end