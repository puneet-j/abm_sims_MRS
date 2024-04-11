clear all;
close all;
clc;
% 
% emb = readmatrix('./GI20I0p5ALLENVS_AGENTS_emb.csv');
% colors_emb = readmatrix('./GI20I0p5ALLENVS_AGENTS_colors.csv',  OutputType="char");
% poses_node = readmatrix('./GI20I0p5ALLENVS_AGENTS_poses.csv',  OutputType="char");
% quals_node = readmatrix('./GI20I0p5ALLENVS_AGENTS_quals.csv');
% global_info_node = readmatrix('./GI20I0p5ALLENVS_AGENTS_global_info.csv');
% num_agents_node = readmatrix('./GI20I0p5ALLENVS_AGENTS_num_agents.csv');

emb = readmatrix('../CDC/20I0p5ALLENVS_AGENTS_emb.csv');
colors_emb = readmatrix('../CDC/20I0p5ALLENVS_AGENTS_colors.csv',  OutputType="char");
poses_node = readmatrix('../CDC/20I0p5ALLENVS_AGENTS_poses.csv',  OutputType="char");
quals_node = readmatrix('../CDC/20I0p5ALLENVS_AGENTS_quals.csv');
global_info_node = readmatrix('../CDC/20I0p5ALLENVS_AGENTS_global_info.csv');
num_agents_node = readmatrix('../CDC/20I0p5ALLENVS_AGENTS_num_agents.csv');


% emb = readmatrix('./NewOneHot_ALLENVS_AGENTS_emb.csv');
% colors_emb = readmatrix('./NewOneHot_ALLENVS_AGENTS_colors.csv',  OutputType="char");
% poses_node = readmatrix('./NewOneHot_ALLENVS_AGENTS_poses.csv',  OutputType="char");
% quals_node = readmatrix('./NewOneHot_ALLENVS_AGENTS_quals.csv');
% global_info_node = readmatrix('./NewOneHot_ALLENVS_AGENTS_global_info.csv');
% num_agents_node = readmatrix('./NewOneHot_ALLENVS_AGENTS_num_agents.csv');

emb = emb(2:end,2:end);
global_info_node = global_info_node(2:end, 2:end);
colors_emb = colors_emb(2:end,2:end);
poses_node = poses_node(2:end, 2:end);
quals_node = quals_node(2:end, 2:end);
num_agents_node = num_agents_node(2:end, 2);

colors = colors_emb;
coords = emb;
% s  = ones(length(emb(:,1)), 1);
quals_node(isnan(quals_node))=0;
global_info_node(isnan(global_info_node))=0;
% poses_node(isnan(poses_node))=1000;

new = ones(size(poses_node,1), size(poses_node,2), 2)*1000;
dist = ones(size(poses_node,1), 4)*1000;
for i=1:length(poses_node(:,1))
    for j=1:4
        p = poses_node(i,j);
        new(i,j,:) = jsondecode(p{1});
        dist(i,j) = round(sqrt(new(i,j,1)^2 + new(i,j,2)^2));
    end
end


A = unique(quals_node, 'rows');
%%
clear new
clear poses_node
clear global_info_node

%%
% c = zeros(length(emb(:,1)), 3);
%%
% figure;
plus = zeros(length(coords(:,1)), 3);
% tri = zeros(length(emb(:,1)), 3);
% circ = zeros(length(emb(:,1)), 3);
c = zeros(length(coords(:,1)), 3);
s  = ones(length(coords(:,1)), 1);
alphas = zeros(length(coords(:,1)), 1);
% counterplus = 1;
% countertri = 1;
% countercirc = 1;
counter  = 1;
% pluses = 
main_counter = 1;
alphas1 = zeros(length(coords(:,1)), 1);
for k = 1:length(s)
    % if ismember(quals_node(k,:),A,'rows')
    q = quals_node(k,:);
    % if ismember(q, A([10 11 12 13 14 15 16 29 30],:),  'rows')
        if length(q(q~=0)) == 2 %&& num_agents_node(k) == 5
            
            if num_agents_node(k) == 5
                % disp(5);
                s(counter) = 40;
            elseif num_agents_node(k) == 10
                % disp(10);
                s(counter) = 80;
            end
            col = colors(k);
            alphas(counter) = 0.5;
            alphas1(counter) = 0.5;
            if col{1} == 'b'
                c(counter,:) = [0.297,0,0.598];
                s(counter) = 1;
                alphas(counter) = 0.1;
                alphas1(counter) = 0.1;
            elseif col{1} == 'r'
                c(counter,:) = [0.8,0,0.8];

            elseif col{1} == 'g'
                c(counter,:) = [0,0.8,0.8];   
            elseif col{1} == 'k'
                c(counter,:) = [0.1,0.1,0.1];
                alphas(counter) = 1.0;
                alphas1(counter) = 1.0;
            end
            plus(counter,:) = coords(k,:);
            counter = counter + 1;
            main_counter  = main_counter + 1;
                % elseif length(q(q~=0)) == 3
                %     tri(k,:) = coords(k,:);
                % elseif length(q(q~=0)) == 4
                %     circ(k,:) = coords(k,:);
            % end
        end
    % end
        % counter = counter + 1;
    % else
    %     disp(quals(k,:));
    % end
end
% c1 = counter+1;
plus = plus(1:counter+1,:);
s = s(1:counter+1);
c = c(1:counter+1,:);
main_counter = main_counter + 1;
alphas1 = alphas1(1:counter+1);

f = figure;
sh1 = scatter3(plus(:,1), plus(:,2), plus(:,3), s, c,'filled' ,'s');%, 'LineWidth',10);
% sh = scatter3(plus(:,1), plus(:,2), plus(:,3), s, c,'filled' ,'s');%, 'LineWidth',10);
% sh.AlphaData = alphas1;
% sh.MarkerFaceAlpha = 'flat';
hold on;

% s.show()

%%
% plus = zeros(length(emb(:,1)), 3);
tri = zeros(length(coords(:,1)), 3);
% circ = zeros(length(emb(:,1)), 3);
c2 = zeros(length(coords(:,1)), 3);
s2  = ones(length(coords(:,1)), 1);
alphas2 = zeros(length(coords(:,1)), 1);
% counterplus = 1;
% countertri = 1;
% countercirc = 1;
counter  = 1;
% pluses = 
for k = 1:length(s2)
    q = quals_node(k,:);
    % if ismember(q, A([1 2 3 4],:),  'rows')
        if length(q(q~=0)) == 3
            
            if num_agents_node(k) == 5
                % disp(5);
                s2(counter) = 40;
            elseif num_agents_node(k) == 10
                % disp(10);
                s2(counter) = 80;
            end
            col = colors(k);
            alphas(main_counter) = 0.5;
            alphas2(counter) = 0.5;
            if col{1} == 'b'
                c2(counter,:) = [0.297,0,0.598];
                s2(counter) = 1;
                alphas(main_counter) = 0.1;
                alphas2(counter) = 0.1;
            elseif col{1} == 'r'
                c2(counter,:) = [0.8,0,0.8];

            elseif col{1} == 'g'
                c2(counter,:) = [0,0.8,0.8];   
            elseif col{1} == 'k'
                c2(counter,:) = [0.1,0.1,0.1];
                alphas(main_counter) = 1.0;
                alphas2(counter) = 1.0;
            end
            tri(counter,:) = coords(k,:);
            counter = counter + 1;
            main_counter = main_counter + 1;
                % elseif length(q(q~=0)) == 3
                %     tri(k,:) = coords(k,:);
                % elseif length(q(q~=0)) == 4
                %     circ(k,:) = coords(k,:);
            % end
        end
    % end
        % counter = counter + 1;
    % else
    %     disp(quals(k,:));
    % end
end

tri = tri(1:counter+1,:);
s2 = s2(1:counter+1);
c2 = c2(1:counter+1,:);
alphas2 = alphas2(1:counter+1);
main_counter = main_counter + 1;
% figure;
sh2 = scatter3(tri(:,1), tri(:,2), tri(:,3), s2, c2, 'filled', '^');%, 'LineWidth',10);
% sh.AlphaData = alphas;
% sh.MarkerFaceAlpha = 'flat';
%%
% plus = zeros(length(emb(:,1)), 3);
% tri = zeros(length(emb(:,1)), 3);
circ = zeros(length(coords(:,1)), 3);
c3 = zeros(length(coords(:,1)), 3);
s3  = ones(length(coords(:,1)), 1);
alphas3 = zeros(length(coords(:,1)), 1);
% counterplus = 1;
% countertri = 1;
% countercirc = 1;
counter  = 1;
% pluses = 
for k = 1:length(s3)
    % if ismember(quals_node(k,:),A,'rows')
    q = quals_node(k,:);
    % if ismember(q, A([7 17 26 27],:),  'rows')
    % if ismember(q, A(7,:),  'rows')
        if length(q(q~=0)) == 4
            
            if num_agents_node(k) == 5
                s3(counter) = 40;
            elseif num_agents_node(k) == 10
                s3(counter) = 80;
            end
            col = colors(k);
            alphas(main_counter) = 0.5;
            alphas3(counter) = 0.5;
            if col{1} == 'b'
                c3(counter,:) = [0.297,0,0.598];
                s3(counter) = 1;
                alphas(main_counter) = 0.1;
                alphas3(counter) = 0.1;
            elseif col{1} == 'r'
                c3(counter,:) = [0.8,0,0.8];

            elseif col{1} == 'g'
                c3(counter,:) = [0,0.8,0.8];   
            elseif col{1} == 'k'
                c3(counter,:) = [0.1,0.1,0.1];
                alphas(main_counter) = 1.0;
                alphas3(counter) = 1.0;
            end
            circ(counter,:) = coords(k,:);
            counter = counter + 1;

        end
    % end
        % counter = counter + 1;
    % else
    %     disp(quals(k,:));
    % end
end
main_counter = main_counter + 1;
circ = circ(1:counter+1,:);
s3 = s3(1:counter+1);
c3 = c3(1:counter+1,:);
alphas3 = alphas3(1:counter+1);

% figure;
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
set(ax,'Fontsize', 20);
view([180, -90, 90]);
box on;

save testvars.mat circ tri plus s c s2 c2 s3 c3 alphas1 alphas2 alphas3



% f.AlphaData = alphas;
% f.MarkerFaceAlpha = 'flat';
% hold on;
% 
% for i = 1:length(alphas)
%     plot3(circ(i,1),  circ(i,2), circ(i,3), 'o', 'MarkerSize',s(i), 'Color',[c(i,:), alphas(i)])
% end
% savefig('testaltogetherGI.fig')
% close all;
% %%
% % plus = zeros(length(emb(:,1)), 3);
% tri = zeros(length(coords(:,1)), 3);
% % circ = zeros(length(emb(:,1)), 3);
% c = zeros(length(coords(:,1)), 3);
% s  = ones(length(coords(:,1)), 1);
% counter  = 1;
% % pluses = 
% for k = 1:length(s)
%     q = quals_node(k,:);
%     if q == A(1,:)
%         if length(q(q~=0)) == 3
% 
%             if num_agents_node(k) == 5
%                 s(counter) = 50;
%             elseif num_agents_node(k) == 10
%                 s(counter) = 100;
%             end
%             col = colors(k);
% 
%             if col{1} == 'r'
%                 c(counter,:) = [0.8,0,0.8];
%             elseif col{1} == 'g'
%                 c(counter,:) = [0,0.8,0.8];   
%             elseif col{1} == 'k'
%                 c(counter,:) = [0.4,0.4,0.4];
%             end
%             if col{1} == 'b'
%                 c(counter,:) = [0.297,0,0.598];
%                 s(counter) = 5;
%                 % if randi(10) < 2
%                 %     % if length(q(q~=0)) == 2
%                 %     %     plus(counter,:) = coords(k,:);
%                 %     % elseif length(q(q~=0)) == 3
%                 %     %     tri(counter,:) = coords(k,:);
%                 %     % elseif length(q(q~=0)) == 4
%                 %     %     circ(counter,:) = coords(k,:);
%                 %     % end
%                 % end
%             end
%                 % if length(q(q~=0)) == 2
%             tri(counter,:) = coords(k,:);
%             counter = counter + 1;
%                 % elseif length(q(q~=0)) == 3
%                 %     tri(k,:) = coords(k,:);
%                 % elseif length(q(q~=0)) == 4
%                 %     circ(k,:) = coords(k,:);
%             % end
%         end
%     end
%         % counter = counter + 1;
%     % else
%     %     disp(quals(k,:));
%     % end
% end
% 
% tri = tri(1:counter,:);
% s = s(1:counter);
% c = c(1:counter,:);
% 
% figure;
% scatter3(tri(:,1), tri(:,2), tri(:,3), s, c, '^');
