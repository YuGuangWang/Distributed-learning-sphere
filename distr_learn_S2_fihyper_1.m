% % distributed learning for noisy data on S^2
clear,clc

p1 = pwd;
addpath(genpath(p1))

% generate spherical design
L = 45; % degree of filtered hyperinterpolation
[w,x] = SD(3*L-1);
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
t0 = tic;
for i = 1:Nev
    ykn = fhS2(L,xev(:,i)'*x); % Fourier kernel
    yfh(i) = sum(ykn.*y.*w);
end
t1 = toc(t0);

% test
ykn1 = fhS2(L,xev(:,end)'*x);
ykn2 = FilteredSphKer_r(2,5,@hyperfilter_R,L,xev(:,end)'*x)/(4*pi);
l2err_kn = norm(ykn1-ykn2)/norm(ykn1);
fprintf(' - l2 error of kernel v1 and v2: %.4e\n',l2err_kn)

% test fh v2
m = 5;
fih = @(t) hyperfilter_R(m,t);
t0 = tic;
yfh2 = zeros(L+1,Nev);
for m = 1:L
    if m==1
        for l=0:1
            yfh2(m+1,:) = yfh2(m+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
        end
    else
        for l = [2*m-2 2*m-1]
            yfh2(m+1,:) = yfh2(m+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
        end
        yfh2(m+1,:) = yfh2(m+1,:) + yfh2(m,:);
    end
end
t2 = toc(t0);

% fh v3
t0 = tic;
yfh3 = zeros(L+1,Nev);
for m = 1:L
    for l = 0:2*m-1
        yfh3(m+1,:) = yfh3(m+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
    end
end
t3 = toc(t0);

% l2 error
yev = fun(xev',k_rbf)';
l2err = norm(yev-yfh)/norm(yev);
l2err2 = norm(yev-yfh2(end,:))/norm(yev);
l2err3 = norm(yev-yfh3(end,:))/norm(yev);
fprintf('Filtered Hyperinterpolation\n')
fprintf(' - filter: s = 5; degree L = %d\n',L)
fprintf(' - l2 error of hyperinterpolation v1: %.4e\n',l2err)
fprintf(' - l2 error of hyperinterpolation v2: %.4e\n',l2err2)
fprintf(' - l2 error of hyperinterpolation v3: %.4e\n',l2err3)
fprintf(' - l2 error of btw v1 and v2: %.4e\n',norm(yfh-yfh2(end,:))/norm(yfh))
fprintf(' - CPU time for hyperinterpolation v1: %.4fs\n',t1)
fprintf(' - CPU time for hyperinterpolation v2: %.4fs\n',t2)
fprintf(' - CPU time for hyperinterpolation v3: %.4fs\n',t3)

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