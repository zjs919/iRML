%% Data Loading
load("data_ent.mat");
load("data_tri.mat");
%%
SignalLength=10000000;
SignalSequence=rand(1,SignalLength);
SignalSequence(SignalSequence>0.5)=1;
SignalSequence(SignalSequence<=0.5)=0;
SignalBpskBit=1-2*SignalSequence;
Bit_tx=SignalBpskBit;
noise_real=normrnd(0,sqrt(0.5),[1 SignalLength]);
noise_image=j*normrnd(0,sqrt(0.5),[1 SignalLength]);
SNR=(-12:2:12);
BER=zeros(1,length(SNR));
for i=1:length(SNR)
    ratio=10^(SNR(i)/10);
    Mesage_rx=sqrt(ratio)*Bit_tx+(noise_real+noise_image);
    Bit_rx=zeros(1,SignalLength);
    Bit_rx(Mesage_rx>0)=1;
    Bit_rx(Mesage_rx<0)=-1;
    SignalBit_rx=0.5*(1-Bit_rx);
    error=xor(SignalSequence,SignalBit_rx);
    BER(i)=(sum(error))/SignalLength;
end
%% Layered Semantics BER Calculation
n = 50000;
index_low = 1:1:length(ent_low(:,1));
random_index_low = index_low(randi(numel(index_low),1,n));
index_mid = 1:1:length(ent_mid(:,1));
random_index_mid = index_mid(randi(numel(index_mid),1,n));
index_high = 1:1:length(ent_high(:,1));
random_index_high = index_high(randi(numel(index_high),1,n));
snrs = -12:2:12;
acc_list_low = zeros(1,length(snrs));
ber_list_low = zeros(1,length(snrs));
ser_list_mid = zeros(1,length(snrs));
ber_list_mid = zeros(1,length(snrs));
ser_list_high = zeros(1,length(snrs));
ber_list_high = zeros(1,length(snrs));
% BER of Low-layer Semantics  
for s = 1:1:length(snrs)
    acc = 0;
for i = 1 :1: n
signal = ent_low(random_index_low(i),:);
noised = awgn(signal, snrs(s),'measured');
temp = ent_low - noised;
mod = [];
    for j = 1:1:length(ent_low(:,1))
    mod(j) = norm(temp(j,:));
    end
[minvalue, min_idex] = min(mod);
if ismember(random_index_low(i),find(mod==min(mod)))
acc = acc+1;
end
end
ser = 1-acc/n;
acc_list_low(s) = ser;
end
ber_list_low = 1-nthroot(1-acc_list_low,8);
% BER of Mid-layer Semantics  
for s = 1:1:length(snrs)
    acc = 0;
for i = 1 :1: n
signal = ent_mid(random_index_mid(i),:);
noised = awgn(signal, snrs(s),'measured');
temp = ent_mid - noised;
mod = [];
    for j = 1:1:length(ent_mid(:,1))
    mod(j) = norm(temp(j,:));
    end
[minvalue, min_idex] = min(mod);
if ismember(random_index_mid(i),find(mod==min(mod)))
acc = acc+1;
end
end
ser = 1-acc/n;
ser_list_mid(s) = ser;
end
ber_list_mid = 1-nthroot(1-ser_list_mid,8);
% BER of High-layer Semantics  
for s = 1:1:length(snrs)
    acc = 0;
for i = 1 :1: n
signal = ent_high(random_index_high(i),:);
noised = awgn(signal, snrs(s),'measured');
temp = ent_high - noised;
mod = [];
    for j = 1:1:length(ent_high(:,1))
    mod(j) = norm(temp(j,:));
    end
[minvalue, min_idex] = min(mod);
if ismember(random_index_high(i),find(mod==min(mod)))
acc = acc+1;
end
end
ser = 1-acc/n;
ser_list_high(s) = ser;
end
ber_list_high = 1-nthroot(1-ser_list_high,8);
%% Semantic Recovery Accuracy with noised signals
n = 50000;
recovery_ratio = 0.3;
index_low = 1:1:length(head_low);
random_index_low = index_low(randi(numel(index_low),1,n));
index_mid = 1:1:length(head_mid);
random_index_mid = index_mid(randi(numel(index_mid),1,n));
index_high = 1:1:length(head_high);
random_index_high = index_high(randi(numel(index_high),1,n));
snrs = [-12:2:12,1000];
acc_list_low = zeros(1,length(snrs));
acc_list_mid = zeros(1,length(snrs));
acc_list_high = zeros(1,length(snrs));
% Recovery Acc. of Low-layer triplets
for s = 1:1:length(snrs)
    acc = 0;
for i = 1 :1: n
signal_h = head_low(random_index_low(i),:);
signal_t = tail_low(random_index_low(i),:);
noised_h = awgn(signal_h, snrs(s),'measured');
noised_t = awgn(signal_t, snrs(s),'measured');
temp = relation_low-(noised_t-noised_h);
norm_temp = norm(relation_low(random_index_low(i),:)-(noised_t-noised_h));
mod = [];
    for j = 1:1:length(head_low)
    mod(j) = norm(temp(j,:));
    end
sorted_mod = sortrows(mod');    
if norm_temp <= sorted_mod(10)
acc = acc+1;
end
end
ser =1-acc/n;
acc_list_low(s) = ser;
end
% Recovery Acc. of Mid-layer triplets
for s = 1:1:length(snrs)
    acc = 0;
for i = 1 :1: n
signal_h = head_mid(random_index_mid(i),:);
signal_t = tail_mid(random_index_mid(i),:);
noised_h = awgn(signal_h, snrs(s),'measured');
noised_t = awgn(signal_t, snrs(s),'measured');
temp = relation_mid-(noised_t-noised_h);
norm_temp = norm(relation_mid(random_index_mid(i),:)-(noised_t-noised_h));
mod = [];
    for j = 1:1:length(head_mid)
    mod(j) = norm(temp(j,:));
    end
sorted_mod = sortrows(mod');    
if norm_temp <= sorted_mod(10)
acc = acc+1;
end
end
ser =1-acc/n;
acc_list_mid(s) = ser;
end
% Recovery Acc. of High-layer triplets
for s = 1:1:length(snrs)
    acc = 0;
for i = 1 :1: n
signal_h = head_high(random_index_high(i),:);
signal_t = tail_high(random_index_high(i),:);
noised_h = awgn(signal_h, snrs(s),'measured');
noised_t = awgn(signal_t, snrs(s),'measured');
temp = relation_high-(noised_t-noised_h);
norm_temp = norm(relation_high(random_index_high(i),:)-(noised_t-noised_h));
mod = [];
    for j = 1:1:length(head_high)
    mod(j) = norm(temp(j,:));
    end
sorted_mod = sortrows(mod');    
if norm_temp <= sorted_mod(10)
acc = acc+1;
end
end
ser =1-acc/n;
acc_list_high(s) = ser;
end
%% Hard Decoding
hard_improvedBER_low = ber_list_low-recovery_ratio.*(1-ber_list_low).*(1-acc_list_low(end)).*ber_list_low;
hard_improvedBER_mid = ber_list_mid-recovery_ratio.*(1-ber_list_mid).*(1-acc_list_mid(end)).*ber_list_mid;
hard_improvedBER_high = ber_list_high-recovery_ratio.*(1-ber_list_high).*(1-acc_list_high(end)).*ber_list_high;
%% Soft Decoding
soft_improvedBER_low = ber_list_low-ber_list_low.*(1-acc_list_low(1:end-1)).*ber_list_low;
soft_improvedBER_mid = ber_list_mid-ber_list_mid.*(1-acc_list_mid(1:end-1)).*ber_list_mid;
soft_improvedBER_high = ber_list_high-ber_list_high.*(1-acc_list_high(1:end-1)).*ber_list_high;