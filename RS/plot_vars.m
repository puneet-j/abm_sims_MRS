clc;
close all;
clear all;

load testvars.mat

f = figure;
sh1 = scatter3(plus(:,1), plus(:,2), plus(:,3), s, c,'filled' ,'s');%, 'LineWidth',10);
hold on;
sh2 = scatter3(tri(:,1), tri(:,2), tri(:,3), s2, c2, 'filled', '^');%, 'LineWidth',10);
sh3 = scatter3(circ(:,1), circ(:,2), circ(:,3), s3, c3,'filled', 'o'); %, 'MarkerFaceAlpha', alphas);

set(sh1, 'AlphaData', alphas1);
set(sh2, 'AlphaData', alphas2);
set(sh3, 'AlphaData', alphas3);

set(sh1, 'MarkerFaceAlpha', 'flat');
set(sh2, 'MarkerFaceAlpha', 'flat');
set(sh3, 'MarkerFaceAlpha', 'flat');
xlabel('x')
ylabel('y')
zlabel('z')
ax = gca;
set(ax,'Fontsize', 16);
% view([-37.5, 30]);
view([210, -120]);
% view([40, -150, 180]);
box on;
clear alphas1 alphas2 alphas3 c c2 c3 s s2 s3 plus tri circ
% saveas(gcf,'myfigure.pdf')
% close all;