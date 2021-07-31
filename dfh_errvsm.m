% % distributed learning for noisy data on S^2
clear,clc

p1 = pwd;
addpath(genpath(p1))

% rotation matrix
A = @(theta) [cos(theta) -sin(theta) 0;sin(theta) cos(theta) 0;0 0 1];
% variance of noise
% sigma2_v = [0.1,0.5,0.8];
sigmav = [0.1,0.01,0.001,0.0001,0];
% mv = [1 10:10:100];
mv = [1 50:50:500]; % number of machines
QH = 'SD';
Lv = 25;
% l2err = zeros(1,length(Lv));
l2errm = zeros(length(sigmav),length(mv));

isig = 0;
for sigma = sigmav
    isig = isig + 1;
    % for each sigma compute distributed filtered hyperinterp
for L=Lv
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
Nev = 100;
[w_ev,xev] = SP(Nev);
xev = xev'; % change size to [3 N]

%% compute distributed filtered hyperinterpolation
j = 0;
for m = mv
    j = j + 1;
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
    theta_i = pi*i/10;
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
l2errm(isig,j) = norm(yev-yfhm(end,:))/norm(yev);
fprintf('**Distributed Filtered Hyperinterpolation\n')
fprintf(' - filter: s = 5; degree L = %d\n',L)
fprintf(' - m: %d, N_QH_i: %d; N_ev = %d\n',m,N1,Nev)
fprintf(' - l2 error of dfh: %.4e\n',l2errm(j))
fprintf(' - CPU time for dfh: %.4fs\n',t2)
fprintf('\n')

end

% %% compute filtered hyperinterpolation
% s = 5;
% fih = @(t) hyperfilter_R(s,t);
% t0 = tic;
% yfh = zeros(L+1,Nev);
% w = 4*pi/length(y);
% for k = 1:L
%     if k==1
%         for l=0:1
%             yfh(k+1,:) = yfh(k+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
%         end
%     else
%         for l = [2*k-2 2*k-1]
%             yfh(k+1,:) = yfh(k+1,:) + fih(l/L)*(2*l+1)/(4*pi)*((w.*y)*legen(l,x'*xev));
%         end
%         yfh(k+1,:) = yfh(k+1,:) + yfh(k,:);
%     end
% end
% t1 = toc(t0);

% % l2 error
% yev = fun(xev')';
% N = length(y);
% l2err(j) = norm(yev-yfh(end,:))/norm(yev);
% fprintf('**Filtered Hyperinterpolation\n')
% fprintf(' - filter: s = 5; degree L = %d\n',L)
% fprintf(' - N_QH: %d; N_ev = %d\n',N,Nev)
% fprintf(' - l2 error of fh: %.4e\n',l2err(j))
% fprintf(' - CPU time for fh: %.4fs\n',t1)
% fprintf('\n')
end

% for i=1:length(Lv)
%     fprintf(' == relative l2 error of fihyp %d: %2.4e\n',Lv(i),l2err(i))
% end

% for i=1:length(Lv)
%     fprintf(' == relative l2 error of distr_fihyp %d: %2.4e\n',Lv(i),l2errm(i))
% end
end

% save data
% save data
subdir = '/';
sv_dat_dir = ['data' subdir];
if ~exist(sv_dat_dir,'dir')
    mkdir(sv_dat_dir)
end
sv_dat = [sv_dat_dir 'errvsm_dfhyp' '_m' num2str(mv(end)) '.mat'];
save(sv_dat)

%% plot
close all
sv_fig_dir = ['figure' subdir];
if ~exist(sv_fig_dir,'dir')
    mkdir(sv_fig_dir);
end
if L == 45
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
pltfunc_rbf_1(y0,x,'',2);
% noisy RBF
pltfunc_rbf_1(y,x,'',4);
end

% % error for filtered hypinterpolation
% fig_err = figure;
% [p,pstr] = fitpow(Lv(2:end),l2err(2:end));
% Lv1 = linspace(Lv(1),Lv(end),100);
% l2err1 = p(1)*Lv1.^(p(2));
% loglog(Lv,l2err,'rp',Lv1,l2err1,'-','MarkerSize',3,'LineWidth',2)
% grid on
% ti = '$l_2$ error of Filtered Hyperinterpolation, $h\in C^5$';
% title(ti,'interpreter','latex')
% pstr1 = ['$' pstr '$'];
% lg = legend('fh',pstr1);
% set(lg,'interpreter','latex')
% % ylim([10^(-9) 10])
% sv_err = [sv_fig_dir 'err_fihyper' num2str(sfi) '_L' num2str(Lv(end)) '.eps'];
% print(fig_err,'-depsc2',sv_err);

% color setting
colors=[0,255,255;  %cyan
        255,0,0;  %red
        153,0,0; % OU Crimson Red
        77,255,166;  % medium aquamarine
        0,179,89; % Green (pigment)
        102,204,25;  %Maya blue
        0,77,230; % Blue (RYB)
        106,0,128; %Patriarch (purple)
        0,213,255]./255; % Capri (blue)

% error for distributed filtered hypinterpolation
nsig = length(sigmav);
pstr = cell(1,nsig);
lgstr = cell(1,nsig);
markers = {'v','h','^','p','o'};
fig_err1 = figure;
for isig = 1:length(sigmav)
    sigma = sigmav(isig);
    l2errm1 = l2errm(isig,:);
    [p,pstr{isig}] = fitpow(mv(1:end),l2errm1(1:end),'m');
    pstr{isig} = ['$' pstr{isig} '$'];
    lgstr{isig} = ['$\sigma = ' num2str(sigma) '$'];
    mv1 = linspace(mv(1),mv(end),100);
    l2err2 = p(1)*mv1.^(p(2));
    semilogy(mv,l2errm1,markers{isig},mv1,l2err2,'-','MarkerSize',3,'LineWidth',2)
    hold on
end
grid on   
ti = sprintf('$l_2$ error of Distributed Filtered Hyperinterpolation, $h\\in C^5$, $n = %d$, %s',L,QH);
title(ti,'interpreter','latex')
lg = legend(lgstr{1},pstr{1},lgstr{2},pstr{2},lgstr{3},pstr{3},lgstr{4},pstr{4},lgstr{5},pstr{5});
set(lg,'interpreter','latex')
% xticks(0:5:30);
% xticklabels({'5','10','15','20','25','30'});
% xlim([0 27])
ylim([10^(-7) 10^2])
xlabel('Number of machines $m$','interpreter','latex')
ylabel('$l_2$ error','interpreter','latex')
hold off
sv_err = [sv_fig_dir 'errvsm_dfhyp' '_m' num2str(mv(end)) '.eps'];
print(fig_err1,'-depsc2',sv_err);
