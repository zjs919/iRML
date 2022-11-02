clear
clc
% Load the data of different numbers of layers. Choose from
% "data_one_layer", "data_two_layers","data_three_layers","data_four_layers","data_five_layers".
load("data_five_layers.mat");

data = data_layer_5_1; % Choose the data from a specific layer to see the numerical results.

emb = data(:,1:10:100);
n = 30000;
index = 1:1:length(emb);
random_index = index(randi(numel(index),1,n));
snrs = [2,3,8,9];
acc_list = zeros(1,length(snrs));
for s = 1:1:length(snrs)
    err = 0;
for i = 1 :1: n
signal = emb(random_index(i),:);
noised_emb = awgn(signal, snrs(s),'measured');
temp = emb - noised_emb;
mod = [];
    for j = 1:1:length(emb)
    mod(j) = norm(temp(j,:));
    end
[minvalue, min_idex] = min(mod);
if min_idex ~= random_index(i)
err = err+1;
end
end


ser = err/n;
acc = 1-ser;
acc_list(s) = acc;
end
