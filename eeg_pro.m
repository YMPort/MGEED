function [time,eeg] = eeg_pro(data_folder)
% convert eeg timestamp to real timestamp
% [eeg] = eeg_pro('expdata\0001')

raw_eeg = importdata([data_folder '\bp.csv']);
elen = size(raw_eeg.data,1);
for i = 1:elen
    d = datetime(raw_eeg.data(i,1)*1000,'ConvertFrom','epochtime','TicksPerSecond',1e3,'Format','HH:mm:ss.SSSSSS');
    t = datestr(d,'HH:MM:SS.FFF');
    time(i).t = t;
end
eeg = raw_eeg.data(:,2:71);
end

