clear all
%% Read spikes for excitatory neurons from the .h5 file
spike_file = [pwd,'/Results/exc_spikes.h5'];
N=10000; % Total number of cells
p_exc=0.8; % pourcentage of excitatory neurons

spk = [];
for i=0:N*p_exc - 1;
    a=hdf5read(spike_file,['/Block_0/segments/Segment_0/spiketrains/SpikeTrain_',num2str(i),'/times']);
    spk = [spk;[a,i*ones(length(a),1)]];
end

%% Read excitatory neuron positions in the network
positions_file = [pwd,'/Results/exc_positions.txt'];
temp = importdata(positions_file);
xpos = temp.data(:,2);
ypos = temp.data(:,3);

%% Construct an spatio-temporal activity map 
Dt = 20;                 % time bin (ms)
Ds = 0.1;                % spatial bin
ITI = 500;               % Intertrial interval (ms)
nb_repet = 60;           % number of repetition
duration = ITI*nb_repet; % Total duration of the simulation
rate = zeros(duration/Dt,1/Ds,1/Ds);

i=1;
for ix=(0:Ds:1-Ds)
    j=1;
    for iy=(0:Ds:1-Ds)
        id = find(xpos>=ix & xpos<ix+Ds & ypos>=iy & ypos<iy+Ds);
        idx_r = find(ismember(spk(:,2), id-1));
        rate(:,i,j) = 1000*hist(spk(idx_r,1),(0:Dt:duration-Dt))/(Dt*length(id));
        j=j+1;
    end
    i=i+1;
end

%% Compare the activity map during stimulation with and without the ChR2-like stimulation

% Compute the activity map during the stimulus duration (~40ms)
act_con = squeeze(nanmean(rate(1:ITI*2/Dt:end-1,:,:) + rate(2:ITI*2/Dt:end-1,:,:)))/2;
act_chr = squeeze(nanmean(rate(1+ITI/Dt:ITI*2/Dt:end-1,:,:) + rate(2+ITI/Dt:ITI*2/Dt:end-1,:,:)))/2;

% Compute linear fit 
idx_f = find(act_chr(:)>1);
[fitobject,gof] = fit(act_con(idx_f),act_chr(idx_f),'poly1');
slope = fitobject.p1;
ord_or= fitobject.p2;

% Plot the comparaison for each pixels
plot(act_con,act_chr,'ko')
hold on
plot([0,40],[0,40]*slope+ord_or,'b')
plot([0,40],[0,40],'r')
xlim([0,40])
ylim([0,40])
xlabel('Control Activity [spk/sec]')
ylabel('ChR2 Activity [spk/sec]')
title(['Linear fit slope = ',num2str(slope)])
axis square
