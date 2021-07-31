function FH = W_FH(QH,eta,funtxt,x_Nm,fis,filterType,d)
% FH = W_FH(QH,eta,funtxt,x_L2,fis,filterType,d)
% computes the filtered hyperinterpolation of degree 2^(j-1), j=0:eta
% with discretisation quadrature QH and filter determined by
% [filterType,filter_m] at points x_L2 for S^d.
%
% Inputs:
% QH -- name of the discretisation quadraure rule; QH = 'GL' or 'SD'
% eta -- highest order of discrete partial sum
% funtxt -- name of target function
% x_L2 -- a point set at which we evaluate
% fis -- smoothness of filter
% filterType -- tyep of the filter used; Typically, we use 'Type I' or
% 'Type II' filter.
% d -- dimension of the sphere
%
% Outputs:
% FH -- filtered hyperinterpolation of fun funtxt at the points set x_L2

if nargin < 7
    d = 2;
end
if nargin < 6
    filterType = 'Type II';
end

switch QH
    case 'GL'
        qH = @GL;
    case 'SD'
        qH = @SD;
end
% generate discretisation quadrature
[w_QH,x_QH] = qH(2^eta*3+1,d);
switch funtxt
    case 'sphHar_ell7k10'
        fun = @SphHar_d2_boost;
        ell = 7;
        k_SH = 10;
        f_val = fun(ell,k_SH,x_QH);
    case 'Franke'
        fun = @Franke;
        f_val = fun(x_QH);
    case 'Wendland_k0'
        fun = @Wendland;
        k_rbf = 0;
        f_val = fun(x_QH,k_rbf);
    case 'Wendland_k1'
        fun = @Wendland;
        k_rbf = 1;
        f_val = fun(x_QH,k_rbf);
    case 'Wendland_k2'
        fun = @Wendland;
        k_rbf = 2;
        f_val = fun(x_QH,k_rbf);
    case 'Wendland_k3'
        fun = @Wendland;
        k_rbf = 3;
        f_val = fun(x_QH,k_rbf);
    case 'Wendland_k4'
        fun = @Wendland;
        k_rbf = 4;
        f_val = fun(x_QH,k_rbf);
    case 'Wendland_k5'
        fun = @Wendland;
        k_rbf = 5;
        f_val = fun(x_QH,k_rbf);
    case 'Wendland_k6'
        fun = @Wendland;
        k_rbf = 6;
        f_val = fun(x_QH,k_rbf);
    case 'RBF_k1_xc6'
        fun = @rbf_multicentre;
        k_rbf = 1;
        f_val = fun(x_QH,k_rbf);
    case 'RBF_k2_xc6'
        fun = @rbf_multicentre;
        k_rbf = 2;
        f_val = fun(x_QH,k_rbf);
    case 'RBF_k3_xc6'
        fun = @rbf_multicentre;
        k_rbf = 3;
        f_val = fun(x_QH,k_rbf);
    case 'RBF_k4_xc6'
        fun = @rbf_multicentre;
        k_rbf = 4;
        f_val = fun(x_QH,k_rbf);
    case 'RBF_k5_xc6'
        fun = @rbf_multicentre;
        k_rbf = 5;
        f_val = fun(x_QH,k_rbf);
    case 'RBF_k6_xc6'
        fun = @rbf_multicentre;
        k_rbf = 6;
        f_val = fun(x_QH,k_rbf);
    case 'caps_a1'
        tf = 27;
        alpha = 2.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'caps_a2'
        tf = 27;
        alpha = 3.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'caps_a3'
        tf = 27;
        alpha = 4.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'caps_a4'
        tf = 27;
        alpha = 5.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'anti_caps_a1'
        tf = 28;
        alpha = 2.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'anti_caps_a2'
        tf = 28;
        alpha = 3.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'anti_caps_a3'
        tf = 28;
        alpha = 4.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'anti_caps_a4'
        tf = 28;
        alpha = 5.8;
        f_val = tfsphere(x_QH',tf,alpha)';
    case 'FrankeCone'
        f_val = FrankeCone(x_QH);
end
switch filterType
    case 'Type I'
        h2 = @filter_h2;
        filterNo = num2str(2*fis+1);
        gamma1 = 2; % truncation const. for the filter function
    case 'Type II'
        h2 = @hyperfilter_R;
end
jvec=0:eta;
FH = zeros(length(jvec),size(x_Nm,1));
rep_Fun_xk = repmat(f_val,[1 size(x_Nm,1)]);
rep_wk = repmat(w_QH,[1 size(x_Nm,1)]);
for i=1:length(jvec)
    j=jvec(i);
    % compute the kernel of filtered hyper. of order j at x_j*xyz'
    switch filterType
        case 'Type I'
        FH_ker = FilteredSphKer_g(d,gamma1,filterNo,h2,2^(j-1),x_QH*x_Nm');
        case 'Type II'            
        FH_ker = FilteredSphKer_r(d,fis,h2,2^(j-1),x_QH*x_Nm');
    end
    FH(i,:) = sum(rep_Fun_xk.*FH_ker.*rep_wk);
    clear FH_ker
end
clear rep_Fun_xj rep_wj