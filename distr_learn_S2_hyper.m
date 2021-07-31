% % distributed learning for noisy data on S^2
clear,clc

p1 = pwd;
addpath(genpath(p1))

% generate spherical design
L = 45; % degree of filtered hyperinterpolation
[w,x] = SD(2*L-1);
w = w'*(4*pi); % change size to [1 N], sum(w) = |S^2| = 4*pi
x = x'; % change size to [3 N]
N = length(w);

% RBF on S^2
ftxt = 'RBF_k3_xc6';
% RBF function
switch ftxt
    case 'RBF_k1_xc6'
        fun = @rbf_multicentre;
        k_rbf = 1;
    case 'RBF_k2_xc6'
        fun = @rbf_multicentre;
        k_rbf = 2;
    case 'RBF_k3_xc6'
        fun = @rbf_multicentre;
        k_rbf = 3;
    case 'RBF_k4_xc6'
        fun = @rbf_multicentre;
        k_rbf = 4;
    case 'RBF_k5_xc6'
        fun = @rbf_multicentre;
        k_rbf = 5;
    case 'RBF_k6_xc6'
        fun = @rbf_multicentre;
        k_rbf = 6;
end
% function values at x, change size to [1 N]
y0 = fun(x',k_rbf)';
% add noise
sigma = 0;
y = y0 + randn(1,N)*sigma;
% filtered hyperinterpolation
Nev = 10;
[w_ev,xev] = SP(Nev);
xev = xev'; % change size to [3 N]
yfh = zeros(1,Nev);
% ykn = fhS2(L,xev'*x); % filtered kernel
% for i = 1:Nev
%     yfh(i) = sum(ykn(i,:).*y.*w);
% end
for i = 1:Nev
    ykn = hyp1(L,xev(:,i)'*x); % Fourier kernel
    yfh(i) = sum(ykn.*y.*w);
end

% test
ykn1 = fhS2(L,xev(:,end)'*x);
ykn2 = FilteredSphKer_r(2,5,@hyperfilter_R,L,xev(:,end)'*x)/(4*pi);
l2err_kn = norm(ykn1-ykn2)/norm(ykn1);
fprintf(' - l2 error of kernel v1 and v2: %.4e\n',l2err_kn)

% test
t0 = tic;
yfh1 = zeros(L+1,Nev);
yfh1(1,:) = 1/(4*pi)*((w.*y)*legen(0,x'*xev));
for l = 1:L
    yfh1(l+1,:) = yfh1(l,:) + (2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
end
t1 = toc(t0);
% l2 error
yev = fun(xev',k_rbf)';
l2err = norm(yev-yfh)/norm(yev);
l2err1 = norm(yev-yfh1(end,:))/norm(yev);
fprintf('Filtered Hyperinterpolation\n')
fprintf(' - filter: s = 5; degree L = %d\n',L)
fprintf(' - l2 error of hyperinterpolation v1: %.4e\n',l2err)
fprintf(' - l2 error of hyperinterpolation v2: %.4e\n',l2err1)
fprintf(' - l2 error of btw v1 and v2: %.4e\n',norm(yfh-yfh1(end,:))/norm(yfh))
fprintf(' - CPU time for hyperinterpolation: %.4fs\n',t1)

%% plot
close all
subdir = '/';
sv_fig_dir = ['figure' subdir];
if ~exist(sv_fig_dir,'dir')
    mkdir(sv_fig_dir);
end
% filter
m = 5;
hfi = @(t) hyperfilter_R(m,t);
t = 0:0.01:2.5;
yfi = hfi(t); 
fig_fi = figure;
plot(t,yfi,'-','LineWidth',2); 
grid on; 
ti = 'Filter $h\in C^5$ for Filtered Hyperinterpolation';
title(ti,'interpreter','latex')
sv_fi = [sv_fig_dir 'fihyper' num2str(m) '.eps'];
print(fig_fi,'-depsc2',sv_fi);

% RBF 
pltfunc_rbf_1(y0,x,'',2);
% noisy RBF
pltfunc_rbf_1(y,x,'',4);

% error for filtered hypinterpolation