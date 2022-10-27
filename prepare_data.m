function [targets, predictors, stftNoisyAudio] ...
    = prepare_data(audio, noise, window, overlap, dftSize)
%Extracts the features of the audio signal in proper dimensions for network training 
%and returns target/predictor pairs as well as the amount of features
% (depending on length of window, which could be made a changable parameter)

%files for test/debug
%audio = audioread("speech_example.wav");
%noise = audioread("WashingMachine-16-8-mono-1000secs.mp3");

%% Windowing conditions

fs = 48000;

%use a segment of 10 ms for windowing
%segmentTime = 0.010; %max recommended size of window
%segmentTime = 0.0054; %size used in denoise MATLAB example
%segmentLength = round(segmentTime*fs);
%overlap = round(segmentLength*0.5);

%window = sqrt(hann(segmentLength, 'periodic'));
%dftSize = length(window);

%the amount of audio features & segments returned for training ML model
featureAmount = length(window)/2 + 1;
segments = 8;

%add the selected noise to the clean audio signals
noisyAudio = add_noise(audio, noise);

%% Power of noise & clean signal

stftNoisyAudio = stft(noisyAudio, fs, ...
    "Window", window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange','onesided');

stftCleanAudio = stft(audio, fs, ...
    "Window", window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange','onesided');

%compute magnitude from STFT of clean + noised signals
cleanSignalFeatures = abs(stftCleanAudio);
noisySignalFeatures = abs(stftNoisyAudio);

%%

powerSpecNoiseAug = [ noisySignalFeatures(:, 1:segments) noisySignalFeatures];

%size of segments
audioSegments = zeros(featureAmount, segments , size(powerSpecNoiseAug,2) - segments + 1);

for ii = 1 : size(powerSpecNoiseAug,2) - segments + 1
    audioSegments(:,:,ii) = powerSpecNoiseAug(:,ii:ii+segments-1);
end

predictors = audioSegments;
targets = cleanSignalFeatures;
