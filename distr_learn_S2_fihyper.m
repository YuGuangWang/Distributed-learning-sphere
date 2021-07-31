% % distributed learning for noisy data on S^2
clear,clc

p1 = pwd;
addpath(genpath(p1))

Lv = 5:3:50;
l2err = zeros(1,length(Lv));
l2errm = zeros(1,length(Lv));
j = 0;
for L=Lv
    j = j + 1;
% generate spherical design
% L = 40; % degree of filtered hyperinterpolation
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

% compute filtered hyperinterpolation
s = 5;
fih = @(t) hyperfilter_R(s,t);
t0 = tic;
yfh = zeros(L+1,Nev);
for k = 1:L
    if k==1
        for l=0:1
            yfh(k+1,:) = yfh(k+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
        end
    else
        for l = [2*k-2 2*k-1]
            yfh(k+1,:) = yfh(k+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
        end
        yfh(k+1,:) = yfh(k+1,:) + yfh(k,:);
    end
end
t1 = toc(t0);

% l2 error
yev = fun(xev',k_rbf)';
l2err(j) = norm(yev-yfh(end,:))/norm(yev);
fprintf('**Filtered Hyperinterpolation\n')
fprintf(' - filter: s = 5; degree L = %d\n',L)
fprintf(' - N_QH: %d; N_ev = %d\n',N,Nev)
fprintf(' - l2 error of fh: %.4e\n',l2err(j))
fprintf(' - CPU time for fh: %.4fs\n',t1)
fprintf('\n')

%% compute distributed filtered hyperinterpolation
m = 10; % number of machines
yfhm = zeros(L+1,Nev);
s = 5;
fih = @(t) hyperfilter_R(s,t);
t0 = tic;
for i=1:m
    Ni = floor(N/m);
    yfh = zeros(L+1,Nev);
    if i<m
        w_i = w((i-1)*Ni+1:i*Ni);
        x_i = x(:,(i-1)*Ni+1:i*Ni);
        y_i = y((i-1)*Ni+1:i*Ni);
    else
        w_i = w((i-1)*Ni+1:end);
        x_i = x(:,(i-1)*Ni+1:end);
        y_i = y((i-1)*Ni+1:end);
    end
    for k = 1:L
        if k==1
            for l=0:1
                yfh(k+1,:) = yfh(k+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w_i.*y_i)*legen(l,x_i'*xev));
            end
        else
            for l = [2*k-2 2*k-1]
                yfh(k+1,:) = yfh(k+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w_i.*y_i)*legen(l,x_i'*xev));
            end
            yfh(k+1,:) = yfh(k+1,:) + yfh(k,:);
        end
    end
    yfhm = yfhm + yfh;
end
yfhm = yfhm/m;
t2 = toc(t0);

% l2 error
yev = fun(xev',k_rbf)';
l2errm(j) = norm(yev-yfhm(end,:))/norm(yev);
fprintf('**Filtered Hyperinterpolation\n')
fprintf(' - filter: s = 5; degree L = %d\n',L)
fprintf(' - N_QH: %d; N_ev = %d\n',N,Nev)
fprintf(' - l2 error of fh: %.4e\n',l2errm(j))
fprintf(' - CPU time for fh: %.4fs\n',t1)
fprintf('\n')

end

for i=1:length(Lv)
    fprintf(' == relative l2 error of fihyp %d: %2.4e\n',Lv(i),l2err(i))
end

for i=1:length(Lv)
    fprintf(' == relative l2 error of distr_fihyp %d: %2.4e\n',Lv(i),l2errm(i))
end


%% plot
close all
subdir = '/';
sv_fig_dir = ['figure' subdir];
if ~exist(sv_fig_dir,'dir')
    mkdir(sv_fig_dir);
end
if L == 45
% filter
k = 5;
hfi = @(t) hyperfilter_R(k,t);
t1 = 0:0.01:2.5;
yfi = hfi(t1); 
fig_fi = figure;
plot(t1,yfi,'-','LineWidth',2); 
grid on; 
ti = 'Filter $h\in C^5$ for Filtered Hyperinterpolation';
title(ti,'interpreter','latex')
sv_fi = [sv_fig_dir 'fihyper' num2str(k) '.eps'];
print(fig_fi,'-depsc2',sv_fi);
% RBF 
pltfunc_rbf_1(y0,x,'',2);
% noisy RBF
pltfunc_rbf_1(y,x,'',4);
end

% error for filtered hypinterpolation
fig_err = figure;
[p,pstr] = fitpow(Lv(2:end),l2err(2:end));
Lv1 = linspace(Lv(1),Lv(end),100);
l2err1 = p(1)*Lv1.^(p(2));
loglog(Lv,l2err,'rp',Lv1,l2err1,'-','LineWidth',2)
grid on
ti = '$l_2$ error of Filtered Hyperinterpolation, $h\in C^5$';
title(ti,'interpreter','latex')
pstr1 = ['$' pstr '$'];
lg = legend('fh',pstr1);
set(lg,'interpreter','latex')
ylim([10^(-9) 10])
sv_err = [sv_fig_dir 'err_fihyper' num2str(k) '_L' num2str(Lv(end)) '.eps'];
print(fig_err,'-depsc2',sv_err);

% error for distributed filtered hypinterpolation
fig_err1 = figure;
[p,pstr] = fitpow(Lv(2:end),l2errm(2:end));
Lv1 = linspace(Lv(1),Lv(end),100);
l2err2 = p(1)*Lv1.^(p(2));
plot(Lv,l2errm,'rp',Lv1,l2err2,'-','LineWidth',2)
grid on
ti = '$l_2$ error of Distributed Filtered Hyperinterpolation, $h\in C^5$';
title(ti,'interpreter','latex')
pstr1 = ['$' pstr '$'];
lg = legend('dfh',pstr1);
set(lg,'interpreter','latex')
% ylim([10^(-9) 10])
sv_err = [sv_fig_dir 'err_dfhyp' num2str(k) '_L' num2str(Lv(end)) '.eps'];
print(fig_err1,'-depsc2',sv_err);
