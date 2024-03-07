clear all;
close all;

%%
clc
%%

emb = readmatrix('./Colored_Res_30ep_0p01_nodes_to_plot.csv');
arr = readmatrix('./Colored_Res_30ep_0p01_arr.csv', ExpectedNumVariables=2, OutputType="char");

%%

x = zeros(length(arr), 1);
y = zeros(length(arr), 1);
z = zeros(length(arr), 1);
c = zeros(length(arr), 3);
s = zeros(length(arr), 1);
a = [];




%%
for k = 2:length(emb)
    e = emb(k,:); % Assuming 'e' is a cell array or a nested array where e{1} is the target
    i = arr(k,2);
    
    % Append data to arrays
    x(k-1) =  e(2);
    y(k-1) =  e(3);
    z(k-1) =  e(4);
    % c(k-1) =  i{1};
    if i{1} == 'b'
        c(k-1,:) =  [0,0,1]'; % This will work if 'i' is numeric. If 'i' is a char or string, handling may need adjustment.
        s(k-1) = 1;
    elseif i{1} == 'k'
        c(k-1,:) =  [1,1,0]';
        s(k-1) = 10;
    elseif i{1} == 'r'
        c(k-1,:) =  [1,0,0]';
        s(k-1) = 10;
    elseif i{1} == 'g'
        c(k-1,:) =  [0,1,0]';
        s(k-1) = 10;
    end
    % Handling alpha values based on condition
    % if strcmp(i, 'b')
    %     a = [a, 0.1];
    % else
    %     a = [a, 1.0];
    % end
end
c(k,:) = [1,1,1]';
s(k) = 1;

%%
red = zeros(length(arr), 3);
blue = zeros(length(arr), 3);
green = zeros(length(arr), 3);
yellow = zeros(length(arr), 3);
% c = zeros(length(arr), 3);
% s = zeros(length(arr), 1);
% a = [];

for k = 2:length(emb)
    e = emb(k,:); % Assuming 'e' is a cell array or a nested array where e{1} is the target
    i = arr(k,2);
    

    % c(k-1) =  i{1};
    if i{1} == 'b'
        % Append data to arrays
        blue(k,:) =  [e(2), e(3), e(4)];
    elseif i{1} == 'k'
        yellow(k,:) =  [e(2), e(3), e(4)];
    elseif i{1} == 'r'
        red(k,:) =  [e(2), e(3), e(4)];
    elseif i{1} == 'g'
        green(k,:) =  [e(2), e(3), e(4)];
    end
end
red = red(2:end,:);
yellow = yellow(2:end,:);
blue = blue(2:end,:);
green = green(2:end,:);


[redC, redV] = convhull(red(:,1), red(:,2), red(:,3), 'Simplify', true);
[greenC, greenV] = convhull(green(:,1), green(:,2), green(:,3), 'Simplify', true);
[yellowC, yellowV] = convhull(yellow(:,1), yellow(:,2), yellow(:,3), 'Simplify', true);
[blueC, blueV] = convhull(blue(:,1), blue(:,2), blue(:,3), 'Simplify', true);

figure;
trimesh(redC, red(:,1), red(:,2), red(:,3), 'FaceVertexCData',[1,0,0], 'FaceColor','none')%, 'EdgeColor','interp')
hold on;
trimesh(greenC, green(:,1), green(:,2), green(:,3), 'FaceVertexCData',[0,1,0], 'FaceColor','none')
trimesh(blueC, blue(:,1), blue(:,2), blue(:,3), 'FaceVertexCData',[0,0,1], 'FaceColor','none')
trimesh(yellowC, yellow(:,1), yellow(:,2), yellow(:,3), 'FaceVertexCData',[0.5,0.5,0], 'FaceColor','none')


%%
figure;
scatter3(x, y, z, s, c, 'o'); % MATLAB handles coloring differently. You may need to adjust 'c' to be numeric or map it to specific colors.
% alpha(a); % Set alpha values directly if applicable