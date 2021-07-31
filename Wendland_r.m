function [phi, Dphi, Iphi] = Wendland_r(r, k, delta, n)
% [phi, Dphi, Iphi] = Wendland_r(r, k, delta, n)
% Evaluate the radial basis function (1/delta^n) \phi_k(r/delta)
% where r is the Euclidean distance for 0 <= r <= 2
% Index k determines which RBF
% Each RBK is normalized so phi(0) = 1 (affects k = 3, 5)
% k = -1 : phi(r) = max(1-r, 0)
% k = 0 : phi(r) = max(1-r, 0)^2 in C^0
% k = 1 : phi(r) = max(1-r, 0)^4 * (4*r+1) in C^2
% k = 2 : phi(r) = max(1-r, 0))^6 * (35*r^2+18*r+3)/3 in C^4
% k = 3 : phi(r) = max(1-r, 0))^8 * (32*r^3+25*r^2+8*r+1) in C^6
% k = 4 : phi(r) = max(1-r, 0))^10 * (429*r^4+450*r^3+210*r^2+50*r+5) in C^8
% k = 5 : phi(r) = max(1-r, 0))^12 * (2048*r^5+2697*r^4+1644*r^3+566*r^2+108*r+9) in C^10
% delta : Scaling factor delta > 0 (Default delta = 1)
% n     : Dimension of points x, y: r = ||x - y||
%       : Default value n = 2 for S^2 in R^3
% If input r is an array then output phi is an array of the same size
% If called with two output arguments, also calcualte derivative of phi
% If call with three output arguments, also calcualte integral of phi on
% [0, 1]

% phi_{v,k) is obtained by repeatedly integrating max(1-r, 0)^v
% k times using the integral operator I(f) = int(s*phi, s=r..1);
% See the Maple worksheet rbf_wendland
% Ref: Holger Wendland "Piecewise polynomial, positive definite and
% compactly supported radial function of minimal degree",
% Advances in Computational Mathemaitcs 4 (1995) 389-396,

% Default to S^2 in R^3
if nargin < 4
    n = 2;
end
if nargin < 3
    delta = 1;
end
if delta ~= 1
    r = r/delta;
end

% Positive par of 1 - r
rp = max(1-r, 0);

% Check if derivative of phi required
CalcDeriv = nargout > 1;
Dphi = [];

% Check if integral of phi from 0 to 1 is required
CalcInt = nargout > 2;
Iphi = [];

% Select which RBF
switch k
    
    case {-1}
        
        % Hat function
        phi = max(1 - r, 0);
        
    case {0}
        
        % Wendland v = 2, k = 0 function in C^0 and H_s(S^2) for s = 3/2
        phi = rp.^2;
        if CalcDeriv
            Dphi = 2*rp;
        end
        if CalcInt
            Iphi = 1/3;
        end
        
        
    case {1}
        
        % Wendland v = 3, k = 1 function in C^2 and H_s(S^2) for s = 5/2
        phi = rp.^4 .* (4*r+1);
        if CalcDeriv
            Dphi = -20*rp.^3 .* r;
        end
        if CalcInt
            Iphi = 1/3;
        end
        
    case {2}
        
        % Wendland v = 4, k = 2 function in C^4 and H_s(S^2) for s = 7/2
        phi = rp.^6 .* ((35*r+18).*r+3)/3;
        if CalcDeriv
            Dphi = -(56/3)*rp.^5 .* r .* (5*r+1);
        end
        if CalcInt
            Iphi = 8/27;
        end
        
    case {3}
        
        % Wendlandv = 5, k = 3 function in C^6 and H_s(S^2) for s = 9/2
        phi = rp.^8 .* (((32*r+25).*r+8).*r+1);
        if CalcDeriv
            Dphi = -22*rp.^7 .* r .* ((16*r+7).*r+1);
        end
        if CalcInt
            Iphi = 4/15;
        end
        
        
    case {4}
        
        % Wendland v = 6, k = 4 function in C^8
        phi = rp.^10 .* ((((429*r+450).*r+210).*r+50).*r+5)/5;
        if CalcDeriv
            Dphi = -(26/5)*rp.^9 .* r .* (((231*r+159).*r+45).*r+5);
        end
        if CalcInt
            Iphi = 128/525;
        end
        
    case {5}
        
        % Wendland v = 7, k = 5 function in C^10
        phi = rp.^12 .* (((((2048*r+2697).*r+1644).*r+566).*r+108).*r+9)/9;
        if CalcDeriv
            Dphi = -(272/9)*rp.^11 .* r .* ((((128*r+121).*r+51).*r+11).*r+1);
        end
        if CalcInt
            Iphi = 128/567;
        end
        
    case {6}
        
        % Wendland v = 8, k = 6 function in C^12
        phi = rp.^14 .* ((((((46189*r+73206).*r+54915).*r+24500).*r+6755).*r+1078).*r+77)/77;
        if CalcDeriv
            Dphi = -(380/77)*rp.^13 .* r .* (((((2431*r+2931).*r+1638).*r+518).*r+91).*r+7);
        end
        if CalcInt
            Iphi = 1024/4851;
        end
                        
    otherwise
        
        fprintf('RBF warning: Unknown case k = %d\n', k);
        phi = [];
        return;
        
end

% Scaling to preserve volumes in R^n
%phi = phi / (delta^n);
%if CalcDeriv
%    Dphi = Dphi / (delta^n);
%end
%if CalcInt
%    Iphi = Iphi / (delta^n);
%end
