function [targets, predictors, stftNoisyAudio] ...
    = prepare_data(audio, fs, noise, window, overlap, dftSize)
%Extracts the features of the audio signal in proper dimensions for network training 
% returns target/predictor pairs and stft of audio with noise added

%files for test/debug
%audio = audioread("speech_example.wav");
%noise = audioread("WashingMachine-16-8-mono-1000secs.mp3");

%% conditions for data format

%the amount of audio features & segments returned for training ML model
featureAmount = length(window)/2 + 1;
segments = 8;
newFs = 8000;

[audio] = mydownsample(audio,fs,newFs);

%add the selected noise to the clean audio signals
noisyAudio = add_noise(audio, noise);

%% Power of noise & clean signal

stftNoisyAudio = stft(noisyAudio, newFs, ...
    "Window", window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange','onesided');

stftCleanAudio = stft(audio, newFs, ...
    "Window", window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange','onesided');

%compute magnitude from STFT of clean + noised signals
cleanSignalFeatures = abs(stftCleanAudio).^2/length(window);
noisySignalFeatures = abs(stftNoisyAudio).^2/length(window);

%%

powerSpecNoiseAug = [ noisySignalFeatures(:, 1:segments) noisySignalFeatures];

%size of segments
audioSegments = zeros(featureAmount, segments , size(powerSpecNoiseAug,2) - segments + 1);

for ii = 1 : size(powerSpecNoiseAug,2) - segments + 1
    audioSegments(:,:,ii) = powerSpecNoiseAug(:,ii:ii+segments-1);
end

predictors = audioSegments;
targets = cleanSignalFeatures;
