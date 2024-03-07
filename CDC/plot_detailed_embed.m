clear all;
close all;


clc
%%

% emb = readmatrix('./AllInfoPlot_emb.csv');
% arr = readmatrix('./AllInfoPlot_arr.csv',  OutputType="char");

emb = readmatrix('./notAllSuccess_GlobalTrained_AllInfoPlot_emb_trained_on_lots_info.csv');
arr = readmatrix('./notAllSuccess_GlobalTrained_AllInfoPlot_arr_trained_on_lots_info.csv',  OutputType="char");
emb = emb(2:end,:);

%%
% 
% x = zeros(length(arr), 1);
% y = zeros(length(arr), 1);
% z = zeros(length(arr), 1);
% c = zeros(length(arr), 3);
% s = zeros(length(arr), 1);
% a = [];




% %%
% for k = 2:length(emb)
%     e = emb(k,:); % Assuming 'e' is a cell array or a nested array where e{1} is the target
%     i = arr(k,2);
% 
%     % Append data to arrays
%     x(k-1) =  e(2);
%     y(k-1) =  e(3);
%     z(k-1) =  e(4);
%     % c(k-1) =  i{1};
%     if i{1} == 'b'
%         c(k-1,:) =  [0,0,1]'; % This will work if 'i' is numeric. If 'i' is a char or string, handling may need adjustment.
%         s(k-1) = 1;
%     elseif i{1} == 'k'
%         c(k-1,:) =  [1,1,0]';
%         s(k-1) = 10;
%     elseif i{1} == 'r'
%         c(k-1,:) =  [1,0,0]';
%         s(k-1) = 10;
%     elseif i{1} == 'g'
%         c(k-1,:) =  [0,1,0]';
%         s(k-1) = 10;
%     end
%     % Handling alpha values based on condition
%     % if strcmp(i, 'b')
%     %     a = [a, 0.1];
%     % else
%     %     a = [a, 1.0];
%     % end
% end
% c(k,:) = [1,1,1]';
% s(k) = 1;
% 
% figure;
% scatter3(x, y, z, s, c, 'o'); % MATLAB handles coloring differently. You may need to adjust 'c' to be numeric or map it to specific colors.
% % alpha(a); % Set alpha values directly if applicable
%%
red = zeros(length(arr), 3);
blue = zeros(length(arr), 3);
green = zeros(length(arr), 3);
yellow = zeros(length(arr), 3);
coords = zeros(length(arr), 3);
c = zeros(length(arr), 3);
colors = zeros(length(arr), 1);
% c = zeros(length(arr), 1);
s = zeros(length(arr), 1);
mkr = s;
pluses = zeros(length(arr), 3);
rounds = zeros(length(arr), 3);
quals = zeros(length(arr), 2);
dists = s;
% a = [];

for k = 2:length(emb)-1
    e = emb(k,:); % Assuming 'e' is a cell array or a nested array where e{1} is the target
    color_column = arr(k,40);
    temp = split(color_column, ',');
    color = temp{2};
    qual1col = split(temp{3}, '[');
    qual1 = str2num(qual1col{2});
    qual2col = arr(k,41);
    temp = split(qual2col, ']');
    qual2 = str2num(temp{1});
    distcol = arr(k,43);
    temp = split(distcol, '(');
    temp = split(temp{2}, ',');
    dist = abs(str2num(temp{1}));
    quals(k,:) = [qual1, qual2];
    dists(k) = dist;
    coords(k,:) = [e(2), e(3), e(4)];
    colors(k) = color;
end

%%
s = s(2:end);  
% c = c(2:end, :);
pluses = pluses(2:end, :);
rounds = rounds(2:end, :);
quals = quals(2:end, :);

dists = dists(2:end);
coords = coords(2:end, :);
%%
[C1, ia1, ic1] = unique(quals(:,1));
[C2, ia2, ic2] = unique(quals(:,2));
arr = unique(cat(1,ia1, ia2));

q_unique = quals(arr, :);
%%
% quals = 
quals_to_test = quals(2,:);
plot_these = zeros(length(arr), 3);
counter = 1;
for k = 1:length(s)
    if ismember(quals(k,:),q_unique,'rows')
        % disp(quals(k,:))
        % disp('is member')
        % mkr(k) = 'o';
        s(counter) = 50;
        if colors(k) == 'b'
            c(counter,:) = [0,0,1];
            s(counter) = 5;
        elseif colors(k) == 'r'
            c(counter,:) = [1,0,0];
        elseif colors(k) == 'g'
            c(counter,:) = [0,1,0];   
        elseif colors(k) == 'k'
            c(counter,:) = [1,1,0];
        end
        plot_these(counter,:) = coords(k,:);
        counter = counter + 1;
    else
        disp(quals(k,:))
    end
end
plot_these = plot_these(2:counter-1, :);
c = c(2:counter-1, :);
s = s(2:counter-1, :);
figure;
grid on;
box on;
% plot3(plot_these(:,1), plot_these(:,2), plot_these(:,3), '+', 'color',c)
scatter3(plot_these(:,1), plot_these(:,2), plot_these(:,3), s, c)

    % if dist == 100
    %     pluses(k, :) = [e(2), e(3), e(4)];
    %     mkr(k) = '+';
    % elseif dist == 200
    %     rounds(k, :) = [e(2), e(3), e(4)];
    %     mkr(k) = 'o';
    % end

    % coords(k,:) = [e(2), e(3), e(4)];

% %     if (max(qual1, qual2) > 0.7) && (color ~= 'b')
% %         s(k) = 50;
% %     elseif color == 'b'
% %         s(k) = 1;
% %     else
% %         s(k) = 100;
% %     end
% % 
% %     % c(k-1) =  i{1};
% %     if color == 'b'
% %         % Append data to arrays
% %         % blue(k,:) =  [e(2), e(3), e(4)];
% %         % s(k) = 1;
% %         c(k) = 'b';
% %         c(k,:) = [0,0,1];
% %     elseif color == 'k'
% %         % yellow(k,:) =  [e(2), e(3), e(4)];
% %         % s(k) = 10;
% %         c(k) = 'y';
% %         c(k,:) = [1,1,0];
% %     elseif color == 'r'
% %         % red(k,:) =  [e(2), e(3), e(4)];
% %         % s(k) = 20;
% %         c(k) = 'r';
% %         c(k,:) = [1,0,0];
% %     elseif color == 'g'
% %         % green(k,:) =  [e(2), e(3), e(4)];
% %         % s(k) = 10;
% %         % c(k) = 'g';
% %         c(k,:) = [0,1,0];
% %     end
% % end
%%
% red = red(2:end,:);
% yellow = yellow(2:end,:);
% blue = blue(2:end,:);
% green = green(2:end,:);
% % % % s = s(2:end);  
% % % % c = c(2:end, :);
% % % % pluses = pluses(2:end, :);
% % % % rounds = rounds(2:end, :);
% color
% coords = coords(2:end, :);


% [redC, redV] = convhull(red(:,1), red(:,2), red(:,3), 'Simplify', true);
% [greenC, greenV] = convhull(green(:,1), green(:,2), green(:,3), 'Simplify', true);
% [yellowC, yellowV] = convhull(yellow(:,1), yellow(:,2), yellow(:,3), 'Simplify', true);
% [blueC, blueV] = convhull(blue(:,1), blue(:,2), blue(:,3), 'Simplify', true);

%%

figure;
% grid on;
% box on;

plot3(pluses(:,1), pluses(:,2), pluses(:,3), '+')%, 'color',c, 'markersize',s);
hold on;
% figure;
plot3(rounds(:,1), rounds(:,2), rounds(:,3), 'o')%, 'color',c, 'markersize',s);
%%

figure;
% grid on;
% box on;

plot3(pluses(:,1), pluses(:,2), pluses(:,3), s, c, '+');
% hold on;
figure;
plot3(rounds(:,1), rounds(:,2), rounds(:,3), s, c, 'o');

% for i=1:length(coords(:,1))
%     plot3(coords(i,1), coords(i,2), coords(i,3), mkr(i), s(i), c(i)); % MATLAB handles coloring differently. You may need to adjust 'c' to be numeric or map it to specific colors.
%     hold on;
% end

%%
% figure;
% trimesh(redC, red(:,1), red(:,2), red(:,3), 'FaceVertexCData',[1,0,0], 'FaceColor','none')%, 'EdgeColor','interp')
% hold on;
% trimesh(greenC, green(:,1), green(:,2), green(:,3), 'FaceVertexCData',[0,1,0], 'FaceColor','none')
% trimesh(blueC, blue(:,1), blue(:,2), blue(:,3), 'FaceVertexCData',[0,0,1], 'FaceColor','none')
% trimesh(yellowC, yellow(:,1), yellow(:,2), yellow(:,3), 'FaceVertexCData',[0.5,0.5,0], 'FaceColor','none')


