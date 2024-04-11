clear all;
close all;
clc;

fl = readmatrix('../ABM_plot_CDC.csv');%,  OutputType="char");
fl = fl(2:end, :);
% fl = sortrows(fl, 8);
f1 = figure(1);
hold on;
plot(fl(:,8), fl(:,3)/6000, 'r--', linewidth=2);
plot(fl(:,8), fl(:,4)/6000, 'r--', linewidth=2);
plot(fl(:,8), fl(:,10), 'k_', markersize=5, linewidth=2);
xlabel('quality diff')
ylabel('time to converge')
ax = gca; 
set(ax, 'FontSize', 20);

% set(f1, 'Fontsize', 20);
f2 = figure(2);
hold on;
plot(fl(:,8), fl(:,6), 'r--', linewidth=2);
plot(fl(:,8), fl(:,7), 'r--', linewidth=2);
plot(fl(:,8), fl(:,10), 'k_', markersize=5, linewidth=2);
xlabel('quality diff')
ylabel('success prob')
ax = gca; 
set(ax, 'FontSize', 20);
% set(f2, 'Fontsize', 20);
for i=1:length(fl(:,1))
    dists = fl(i,9);
    if dists == 100.0
        figure(1);
        plot(fl(i,8), fl(i,2)/6000, 'r+', markersize=8, linewidth=2);
        figure(2);
        plot(fl(i,8), fl(i,5), 'r+', markersize=8, linewidth=2);
    elseif dists == 150.0
        figure(1);
        plot(fl(i,8), fl(i,2)/6000, 'ro', markersize=8, linewidth=2);
        figure(2);
        plot(fl(i,8), fl(i,5), 'ro', markersize=8, linewidth=2);
    elseif dists == 200.0
        figure(1);
        plot(fl(i,8), fl(i,2)/6000, 'r^', markersize=8, linewidth=2);
        figure(2);
        plot(fl(i,8), fl(i,5), 'r^', markersize=8, linewidth=2);
    end

end
figure(1);
h1 = plot(NaN,NaN,'r--', 'LineWidth', 2); % Dummy plot for the first legend entry
h2 = plot(NaN,NaN,'k_', 'LineWidth', 2); % Dummy plot for the second legend entry
h3 = plot(NaN,NaN,'r+', 'LineWidth', 2); % Dummy plot for the second legend entry
h4 = plot(NaN,NaN,'ro', 'LineWidth', 2); % Dummy plot for the second legend entry
h5 = plot(NaN,NaN,'r^', 'LineWidth', 2); % Dummy plot for the second legend entry
legend([h1, h2, h3, h4, h5], {'I-Q Range', 'Max Qual', 'dist=100', 'dist=150', 'dist=200'});
figure(2);
h1 = plot(NaN,NaN,'r--', 'LineWidth', 2); % Dummy plot for the first legend entry
h2 = plot(NaN,NaN,'k_', 'LineWidth', 2); % Dummy plot for the second legend entry
h3 = plot(NaN,NaN,'r+', 'LineWidth', 2); % Dummy plot for the second legend entry
h4 = plot(NaN,NaN,'ro', 'LineWidth', 2); % Dummy plot for the second legend entry
h5 = plot(NaN,NaN,'r^', 'LineWidth', 2); % Dummy plot for the second legend entry
legend([h1, h2, h3, h4, h5], {'I-Q Range', 'Max Qual', 'dist=100', 'dist=150', 'dist=200'});

% legend('d=100', 'd=150', 'd=200')