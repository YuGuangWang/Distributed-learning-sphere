function y = legen(n,z)
% evaluates Legendre polynomial by recurrence formula
% INPUTS:
%  n - degree of Legendre polynomial
%  z - variable of Legendre polynomial
% OUTPUTS:
%  y - function values of Legendre polynomial of degree l at z

if n==0
    p0 = ones(size(z));
    y = p0;
elseif n==1
    p1 = z;
    y = p1;
else
    p0 = ones(size(z));
    p1 = z;
    for l = 2:n
        l1 = l-1;
        Al1 = (2*l1+1)/(l1+1);
        Cl1 = -l1/(l1+1);
        pl = Al1*z.*p1 + Cl1*p0;
        p0 = p1;
        p1 = pl;
    end
    y = pl;
end