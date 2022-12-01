% Make prediction about clean speech with a trained model (only works with one sample for now)

%TODO: maybe make this a function to be called in other scripts?

clear,
[noise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
dataset = fullfile("voiceData","commonvoice"); 

%% Load a selection of data samples 

trainDataVoices = audioDatastore(fullfile(dataset,"test"),"IncludeSubfolders",true);

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

T = tall(audioData); 

%preset
segments = 8;

segmentLength = 1024;
%segmentLength= 256;

window = sqrt(hann(segmentLength, 'periodic'));
overlap = round(segmentLength*0.5);

dftSize = length(window);
features = length(window)*0.5;

%%
[targets, predictors, noisedStft] = ...
    cellfun(@(x)prepare_data(x, ...
        fs, ...
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
%normalisation of data

predictors = cat(3,predictors{:});
noisyMean = mean(predictors(:));
noisyStd = std(predictors(:));
predictors(:) = (predictors(:) - noisyMean)/noisyStd;

targets = cat(2,targets{:});
cleanMean = mean(targets(:));
cleanStd = std(targets(:));
targets(:) = (targets(:) - cleanMean)/cleanStd;

predictors = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targets = reshape(targets,1,1,size(targets,1),size(targets,2));

%split into training and validation

trainPredictors = predictors(1:features,:,:,:);
trainTargets = targets(:,:,1:features,:);

%% prediction
%load a network

%load denoiseDNN2Conv.mat
%load tinyNet1.mat

predictionData = trainPredictors;
%speech = predict(denoiseNetFullyConvolutional,predictionData);
%speech = predict(liteConvNet,predictionData);
speech = predict(convNetEnDecode, predictionData);

%% convert prediction into audio signal

noisePhase = angle(noisedStft{1});

speech(:) = cleanStd*speech(:) + cleanMean;

%beware of dimensions - for some reason speech might be -1 too short
speech = squeeze(speech(:,:,:,1:end-1)).* exp(1j*noisePhase(1:end-1,:)); 

speech = [conj(speech(end-1:-1:2,:)); speech];

%NB: window size/overlap needs to match 1st dimension on speech
speechReconstruct = istft(speech, fs, ...
    "Window", window, "OverlapLength", overlap, ... 
    "FFTLength", dftSize, "ConjugateSymmetric", true);

figure, plot(speechReconstruct);

%%

%get rid of the rather significant amplitude error
speechAug = speechReconstruct(100:end-100);

figure, plot(speechAug)
figure, plot(audioData{1})

figure, spectrogram(speechAug, window, overlap, dftSize, 'yaxis')
figure, spectrogram(audioData{1}, window, overlap, dftSize, 'yaxis')

