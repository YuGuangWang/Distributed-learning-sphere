function [w,y] = SD(L,d)
% [w,y] = SD(L,d)
% computes the symmetric spherical design of Rob
%
% Inputs:
% L -- degree for which the SD quadrature is exact
% d -- dimension of sphere; by default d = 2
%
% Outputs:
% w -- weights of quadrature rule, size(w) = [Num. weights, 1]
% y -- nodes of quadrature rule, size(y)=[Num. points, 3]

if nargin < 2
    d = 2;
end

if mod(L,2)==0
    L = L+1;
%     disp('L should be odd.');
%     return;
end

loadfpath = 'Points/SSD/';
if mod(L,2)~=0
    if L<10
        Ltxt = ['00' num2str(L)];
    elseif L<100
        Ltxt = ['0' num2str(L)];
    else
        Ltxt = num2str(L);
    end
    ld = [loadfpath 'ss' Ltxt '.mat'];
    load(ld,'y');
end
w = 1/size(y,1)*ones(size(y,1),1);