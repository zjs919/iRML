emb = load("entity2vec.bern");
n = 30000;
index = 1:1:length(emb);
random_index = index(randi(numel(index),1,n));
ser_list_m = zeros(1,11);
for s = -5  : 1: 5
    err = 0;
for i = 1 :1: n
signal = emb(random_index(i),:);
noised_emb = awgn(signal, s,'measured');
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
ser_list_m(s+6) = ser;
end
