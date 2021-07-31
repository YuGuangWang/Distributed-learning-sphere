% compute the filtered hyperinterpolation on sphere

clear,clc
close all

% subdir =  '\'; % for windows
subdir = '/'; % for linux

sv_dir = ['FH' subdir];
if ~exist(sv_dir,'dir')
    mkdir(sv_dir);
end

% parameters
% d = 2; % dimension of sphere
neord = 2; % order of needlets
N_power = 4; %%%%%%
N_Nm = 10^N_power; % number of nodes for evaluation
% filterType = 'Type I';
% filterType = 'Type II';
fis = 5;
QH = 'SD'; %%%%%%
QNm = 'SP'; %%%%%%
switch QH
    case 'GL'
        qH = @GL;
    case 'SD'
        qH = @SD;
end
switch QNm
    case 'GL'
        qNm = @GL;
    case 'SP'
        qNm = @SP;
end

% funtxt = 'sph harm';
% funtxt = 'RBF_k1_xc6';
% funtxt = 'RBF_k2_xc6';
% funtxt = 'RBF_k3_xc6';
% funtxt = 'RBF_k4_xc6';
% funtxt = 'RBF_k5_xc6';
% funtxt = 'RBF_k6_xc6';

% funtxt_ce = {'RBF_k1_xc6','RBF_k2_xc6','RBF_k3_xc6','RBF_k4_xc6',...
%     'RBF_k5_xc6','RBF_k6_xc6'};
funtxt_ce = {'RBF_k2_xc6'};

% generate spiral points on S^2
xc = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
x_Nm = qNm(N_Nm);
x_Nm = [x_Nm' xc']';
% x_Nm = xc;

for i = 1:length(funtxt_ce)
    funtxt = funtxt_ce{i};

% compute the filtered hyperinterpolation for Y_{ell,k}
FH = W_FH(QH,neord,funtxt,x_Nm,fis);

% compute pointwise error
    switch funtxt
        case 'sphHar_ell7k10'
            fun = @SphHar_d2_boost;
            ell = 7;
            k_SH = 10;
            f_Nm = fun(ell,k_SH,x_Nm);
        case 'RBF_k1_xc6'
            fun = @rbf_multicentre;
            k_rbf = 1;
            f_Nm = fun(x_Nm,k_rbf);
        case 'RBF_k2_xc6'
            fun = @rbf_multicentre;
            k_rbf = 2;
            f_Nm = fun(x_Nm,k_rbf);
        case 'RBF_k3_xc6'
            fun = @rbf_multicentre;
            k_rbf = 3;
            f_Nm = fun(x_Nm,k_rbf);
        case 'RBF_k4_xc6'
            fun = @rbf_multicentre;
            k_rbf = 4;
            f_Nm = fun(x_Nm,k_rbf);
        case 'RBF_k5_xc6'
            fun = @rbf_multicentre;
            k_rbf = 5;
            f_Nm = fun(x_Nm,k_rbf);
        case 'RBF_k6_xc6'
            fun = @rbf_multicentre;
            k_rbf = 6;
            f_Nm = fun(x_Nm,k_rbf);
    end
    f_Nm = f_Nm';
    f_Nm_repmat = repmat(f_Nm,[size(FH,1) 1]);
    FH_err = abs(FH - f_Nm_repmat);
    FH_err_max = max(FH_err,[],2);
    % save mat data
    sv = [sv_dir 'f' funtxt '_FH' '_QH' QH '_fis' num2str(fis)...
        '_Ne' num2str(N_power) '_j' num2str(neord) '.mat'];
    save(sv, 'FH','FH_err','f_Nm','x_Nm');   
end