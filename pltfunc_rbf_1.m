function [fig1,fig2] = pltfunc_rbf_1(F, X, tstr, ifig, scale, ipr)
% [K, fh1, fh2, fig1,fig2] = pltfunc(F, X, tstr, ifig, scale, ipr)
% Plot the function F at the m points X on the unit sphere S^2 in R^3
% Title string tstr using figure windows ifig and ifig+1 and scaling scale
% are all optional

% Default arguments
if nargin < 6
    ipr = 0;
end
if nargin < 5
    scale = 0.5;
end
if nargin < 4
    ifig = 1;
end
if nargin < 3
    tstr = [];
end

% Default view
%vw = [45, 10];
% vw = [-60 25];

% Use Matlab 6 routine based on QHULL
K = convhulln(X');

X1=[X(1,K(:,1)); X(1,K(:,2)); X(1,K(:,3))];
Y1=[X(2,K(:,1)); X(2,K(:,2)); X(2,K(:,3))];
Z1=[X(3,K(:,1)); X(3,K(:,2)); X(3,K(:,3))];
C = F(K');

if ifig > 0
    fig1 = figure(ifig); clf;
    fh1 = patch(X1, Y1, Z1, C);
    colormap(jet(255));
    view(90,0);
    axis vis3d
    axis equal tight
    view([1 1 1]);
    grid off
    set(gca, 'Visible', 'off')
%     colorbar('SouthOutside');
    cbh1 = colorbar('Location','EastOutside');
    if ipr > 10
        hold on
        plot3(X(1,:), X(2,:), X(3,:), 'k.', 'MarkerSize', 16);
        hold off
    else
        set(fh1, 'EdgeColor', 'none')
    end
    title(tstr);
end

[Fmax, imax] = max(F);
%xmax = X(:,imax);
[Fmin, imin] = min(F);
%xmin = X(:,imin);
fprintf('Minimum function value = %.6f, Maximum function value = %.6f\n', Fmin, Fmax);

FS = 1 + (scale/(Fmax-Fmin))*(C-Fmin);
%FS = C;

fig2 = figure(abs(ifig)+1);
clf;
fh2 = patch(X1.*FS, Y1.*FS, Z1.*FS, C);
set(fh2, 'EdgeColor', 'none');
%colormap(jet(255));
colormap(jet(1023));
%colormap(gray(1023));
% cbh = colorbar('SouthOutside');
cbh2 = colorbar('Location','EastOutside');
%cbh = colorbar;
cbp = get(cbh2, 'Position');
% position = [left bottom width height]
%cbp(4) = 0.5*cbp(4);
% cbp(3) = 1*cbp(3);
% set(cbh, 'Position', cbp);
%set(cbh, 'FontSize', 6);
%view(90,0);
axis vis3d
axis equal tight
% switch ipr
%     case 1
%     otherwise
% % set tight axis
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset; 
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% 
% % set page size is equal to figure size
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% end
%view(xmax);
    AZ=154;
    EL=35;
    view([AZ EL]); % modified view
%grid on
axis off
title(tstr);
