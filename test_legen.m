% test legen.m
l = 110;
z = 0:10^(-6):1;
t0 = tic;
y1 = legen(l,z);
t1 = toc(t0);
t0 = tic;
y2_1 = legendre(l,z);
t2 = toc(t0);
y2 = y2_1(1,:);
% y1,y2
l2err = norm(y1-y2)/norm(y2);
fprintf(' - l2 error btw two methods: %.4e\n',t1)
fprintf(' - CPU time of recurrence method: %.4f\n',t1)
fprintf(' - CPU time of legendre: %.4f\n',t2)