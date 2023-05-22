function [ omg ] = omg_AR( AR_path )
%emg_AR( 'expdata\0001\emteqAR.mat' )
load(AR_path)

omg = [Nav_XA' Nav_YA' Nav_XB' Nav_YB' Nav_XC' Nav_YC' Nav_XD' Nav_YD' Nav_XE' Nav_YE' ...
    Nav_XF' Nav_YF' Nav_XG' Nav_YG' Nav_XH' Nav_YH'];

omg = [double(omg) prox2Mm(Prox_A') prox2Mm(Prox_B') prox2Mm(Prox_C') prox2Mm(Prox_D')];
fs = 1000; 
fr = [30, 450]; 
for k = 1:size(omg, 2)
    omg(:, k) = detrend(omg(:, k)); % signal detrending
    omg(:, k) = detrend(omg(:, k), 'constant'); % baseline correction
    for l = 1:7
        % apply the notch filter to remove 50Hz and their harmonics up to 350Hz
        f0 = l*0.05;
        filt_notch = fdesign.notch(4, f0, 10); 
        h_notch = design(filt_notch);
        omg(:, k) = filter(h_notch, omg(:, k));
    end
    %emg(:, k) = fdesign.bandpass(emg(:, k), fs, fr(1), fr(2)); % apply the bandpass filter
    omg(:, k) = bandpass(omg(:, k), fr, fs);
end
end

function Mm = prox2Mm( Prox )
a = 31.4; b = -0.001399; c = 14.14; d = -9.446e-05;
dProxDivided = 8;
dProx = double(Prox) * dProxDivided;
Mm = a * exp(b * dProx) + c * exp(d * dProx);
end