function [w,y] = SP(N,d)
% Y = SP(N)
% generates N-configuration spiral points on the sphere S^2.
%
% Input:
% N -- number of points
%
% Outputs:
% Y -- N spiral nodes, size(Y) = [N 3].

if nargin < 2
    d = 2;
end

subdir = '\'; % for windows
% subdir = '/'; % for linux

if N == 1e6
    % boost for L==1000
    Quadr = 'SP';
    load_dir = ['Points' subdir Quadr subdir]; 
    ldpath = [load_dir 'd' num2str(d) '_' Quadr 'Ne' '6'];
    load(ldpath);
else
    if d == 2
        j=(1:N)';
        zz = 1-(2*j-1)/N;
        theta = acos(zz);
        phi = mod(1.8*sqrt(N)*theta,2*pi);
        xx = sin(theta).*cos(phi);
        yy = sin(theta).*sin(phi);
        y = [xx yy zz];
    end
end
w = ones(N,1)/N;