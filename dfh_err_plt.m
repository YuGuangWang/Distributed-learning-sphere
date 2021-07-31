% % distributed learning for noisy data on S^2
clear,clc

p1 = pwd;
addpath(genpath(p1))

% rotation matrix
A = @(theta) [cos(theta) -sin(theta) 0;sin(theta) cos(theta) 0;0 0 1];
% variance of noise
sigma = [0.1];
QH = 'SD';
L= 25;

% generate spherical design
% L = 40; % degree of filtered hyperinterpolation
[w1,x1] = SD(3*L-1);
w1 = w1'*(4*pi); % change size to [1 N], sum(w) = |S^2| = 4*pi
N1 = length(w1);

% RBF on S^2
ftxt = 'RBF_k3_xc6';
% RBF function
switch ftxt
    case 'RBF_k1_xc6'
        k_rbf = 1;
    case 'RBF_k2_xc6'
        k_rbf = 2;
    case 'RBF_k3_xc6'
        k_rbf = 3;
    case 'RBF_k4_xc6'
        k_rbf = 4;
    case 'RBF_k5_xc6'
        k_rbf = 5;
    case 'RBF_k6_xc6'
        k_rbf = 6;
end
fun = @(x) rbf_multicentre(x,k_rbf);
% point for evaluation
Nev = 1000;
[w_ev,xev] = SP(Nev);
xev = xev'; % change size to [3 N]

%% compute distributed filtered hyperinterpolation
m = 100; % number of machines
yfhm = zeros(L+1,Nev); % function values of distributed filtered hyperinterp
s = 5;
fih = @(t) hyperfilter_R(s,t);
% x = [];
% y = [];
t0 = tic;
for i=1:m
    yfh = zeros(L+1,Nev);
    Ni = N1;
    w_i = w1;
    theta_i = pi*i/m;
    x_i = A(theta_i)*x1';
    % function values at x, change size to [1 N]
    yi1 = fun(x_i')';
    % add noise
    y_i = yi1 + randn(1,Ni)*sigma;
    % compute data at all machines
%     x = [x x_i];
%     y = [y y_i];
    % compute distributed filtered hyperinterp
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
yev = fun(xev')';
l2errm = norm(yev-yfhm(end,:))/norm(yev);
fprintf('**Distributed Filtered Hyperinterpolation\n')
fprintf(' - filter: s = 5; degree L = %d\n',L)
fprintf(' - m: %d, N_QH_i: %d; N_ev = %d\n',m,N1,Nev)
fprintf(' - l2 error of dfh: %.4e\n',l2errm)
fprintf(' - CPU time for dfh: %.4fs\n',t2)
fprintf('\n')

% save data
subdir = '/';
sv_dat_dir = ['data' subdir];
if ~exist(sv_dat_dir,'dir')
    mkdir(sv_dat_dir)
end
sv_dat = [sv_dat_dir 'pntwise_err_dfh' '_L' num2str(L) '.mat'];
save(sv_dat)

%% plot
close all
subdir = '/';
sv_fig_dir = ['figure' subdir];
if ~exist(sv_fig_dir,'dir')
    mkdir(sv_fig_dir);
end
% filter
sfi = 5;
hfi = @(t) hyperfilter_R(sfi,t);
t1 = 0:0.01:2.5;
yfi = hfi(t1); 
fig_fi = figure;
plot(t1,yfi,'-','LineWidth',2); 
grid on; 
ti = 'Filter $h\in C^5$ for Filtered Hyperinterpolation';
title(ti,'interpreter','latex')
sv_fi = [sv_fig_dir 'fihyper' num2str(sfi) '.eps'];
print(fig_fi,'-depsc2',sv_fi);
% RBF 
[~,fig_rbf] = pltfunc_rbf_1(yev,xev,'',2);
sv_rbf = [sv_fig_dir 'pntwise_rbf' num2str(sfi) '_L' num2str(L) '.png'];
print(fig_rbf,'-dpng','-r300',sv_rbf);

% noisy RBF
yev_noisy = yev + rand(1,numel(yev))*0.1;
[~,fig_rbf] = pltfunc_rbf_1(yev_noisy,xev,'',2);
sv_rbf = [sv_fig_dir 'pntwise_rbf_noisy' num2str(sfi) '_L' num2str(L) '.png'];
print(fig_rbf,'-dpng','-r300',sv_rbf);

% dfh
[~,fig_noisy_rbf] = pltfunc_rbf_1(yfhm(end,:),xev,'',4);
sv_noisy_rbf = [sv_fig_dir 'pntwise_dfh' num2str(sfi) '_L' num2str(L) '.png'];
print(fig_noisy_rbf,'-dpng','-r300',sv_noisy_rbf);

% error
[~,fig_err] = pltfunc_rbf_1(yev-yfhm(end,:),xev,'',6.'',1);
sv_err = [sv_fig_dir 'pntwise_err' num2str(sfi) '_L' num2str(L) '.png'];
print(fig_err,'-dpng','-r300',sv_err);

