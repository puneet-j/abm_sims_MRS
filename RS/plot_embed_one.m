clc;
clear all;
close all;
% vals=py.numpy.load('arrdict.npy');
%%
% len = py.numpy.shape(vals);

%%
% double(len(1))

%%
% vals{1}
arr = load('test.mat');

%%
f = figure(1);
for i=1:length(arr)
    % val = vals(i);
    x = arr.x(i);
    y = arr.y(i);
    z = arr.z(i);
    c = arr.col(i);
    sz = arr.sz(i)*10;
    hold on;
    % scatter3(arr.x, y, z, sz, c, 'filled');
end
%%
new = zeros(3,length(arr.col));
for i=1:length(arr.col)
    if arr.col(i) == 'b'
        new(:,i) = [0, 0, 1];
    elseif arr.col(i) == 'k'
        new(:,i) = [0, 0, 0];
    elseif arr.col(i) == 'g'
        new(:,i) = [0, 1, 0];
    elseif arr.col(i) == 'r'
        new(:,i) = [1, 0, 0];
    end

end



%%
scatter3(arr.x, arr.y, arr.z, arr.sz, new.', 'o', 'filledww');

