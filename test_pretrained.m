% Make prediction with a trained model (only works with one sample for now)

% add audio data

clear,

[noise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
dataset = fullfile("voiceData","commonvoice"); 

%% Load a selection of data samples 

trainDataVoices = audioDatastore(fullfile(dataset,"train"),"IncludeSubfolders",true);

samplesIncluded = 1; %load a section of the dataset
trainDataVoices = shuffle(trainDataVoices);
trainDataVoices = subset(trainDataVoices,1:samplesIncluded);
fs = 48000;

%% Sample rate conversion for noise signal
%convert the sample rate of the noise signal to be the same as audioData
if fsNoise ~= fs
 
    inputFs = fsNoise;
    bandwidth = 7920;
    decimationFactor = inputFs/fs;
    L = floor(numel(noise)/decimationFactor);
    noise = noise(1:decimationFactor*L);

    sampleRateConv = dsp.SampleRateConverter('InputSampleRate', inputFs,...
        'OutputSampleRate', fs, 'Bandwidth', bandwidth);

    noise = sampleRateConv(noise);
end

%% prepare data as tall arrays

audioData = readall(trainDataVoices);

%preset 
segments = 8;

%NB: tall is a keyword that by default starts a parallel pool
%when using a small data sample it runs faster with this setting deactivated
T = tall(audioData); 

segmentTime = 0.00533; %in seconds = windowLength at 256 samples
segmentLength = round(segmentTime*fs);
window = sqrt(hann(segmentLength, 'periodic'));
overlap = round(segmentLength*0.5);

dftSize = length(window);
%featureAmount = length(window)/2;

%%
[targets, predictors, noisedStft] = ...
    cellfun(@(x)prepare_data(x, ...
        noise, ...
        window, ...
        overlap, ...
        dftSize...
        ), ...
    T, ...
    UniformOutput=false ...
    ); 

[targets, predictors, noisedStft] = gather(targets,predictors,noisedStft);

%%
%normalisation of data...

predictors = cat(3,predictors{:});
noisyMean = mean(predictors(:));
noisyStd = std(predictors(:));
predictors(:) = (predictors(:) - noisyMean)/noisyStd;

targets = cat(2,targets{:});
cleanMean = mean(targets(:));
cleanStd = std(targets(:));
targets(:) = (targets(:) - cleanMean)/cleanStd;
%reshape

predictorsR = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targetsR = reshape(targets,1,1,size(targets,1),size(targets,2));

%split into training and validation

inds = randperm(size(predictorsR,4));
L = floor(0.90*size(predictorsR,4));

trainPredictors = predictorsR(:,:,:,:);
trainTargets = targetsR(:,:,:,:);

%% prediction

%load CNN and make prediction
load denoiseDNN2Conv.mat

predictionData = trainPredictors;
speech = predict(denoiseNetFullyConvolutional,predictionData);

%% convert prediction into audio signal

noisePhase = angle(noisedStft{1});

speech(:) = cleanStd*speech(:) + cleanMean;

%beware of dimensions - for some reason speech might be -1 too short
speech = squeeze(speech(:,:,:,1:end-1)).* exp(1j*noisePhase); 

speech = [conj(speech(end-1:-1:2,:)); speech];

%NB: window size/overlap needs to match 1st dimension on speech
speechReconstruct = istft(speech, fs, ...
    "Window", window, "OverlapLength", overlap, ... 
    "FFTLength", dftSize, "ConjugateSymmetric", true);

figure, plot(speechReconstruct);
