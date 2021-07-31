function YJ=JacobiRecursiveT2(a,b,L,T)
% Calculate values of Jacobi polynomial of degree L at T
if L==0;
    YJ=ones(size(T));
elseif L==1;
    YJ=(a-b)/2+(a+b+2)/2*T;
else
%     pM=zeros(L+1,length(T));
    pMisb1=ones(size(T));
    pMi=(a-b)/2+(a+b+2)/2*T;
    for i=2:L
        c=2*i+a+b;
        tmppMisb1=pMi;
        pMi=((c-1)*c*(c-2)*T.*pMi+(c-1)*(a^2-b^2)*pMi...
            -2*(i+a-1)*(i+b-1)*c*pMisb1)/(2*i*(i+a+b)*(c-2));
        pMisb1=tmppMisb1;
    end
    YJ=pMi;
end
