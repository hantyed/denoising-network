clear,
%prepare data and build a "small" convolution model

%load noise sample and all data from commonvoice dataset
[washNoise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
dataset = fullfile("voiceData","commonvoice"); 

%% Load a selection of data samples 

trainDataVoices = audioDatastore(fullfile(dataset,"train"),"IncludeSubfolders",true);

samplesIncluded = 100; %loads only 100 samples to keep my computer alive and well
trainDataVoices = shuffle(trainDataVoices);
trainDataVoices = subset(trainDataVoices,1:samplesIncluded); 

%% Play example of clean audio

[audio,audioInfo] = read(trainDataVoices);
sound(audio,audioInfo.SampleRate)

%% Windowing conditions

fs = 44100;

segmentTime = 0.015;
segmentLength = round(segmentTime*fs);
overlap = round(segmentLength*0.5);
dftSize = 1024; %CHECK: should this be size of window*0.5 - 1 ??

window = sqrt(hann(segmentLength, 'periodic'));

%% Power of noise & clean signal

stftNoise = stft(washNoise, fsNoise, ...
    "Window", window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange','onesided');

powerSpecNoise = abs(stftNoise).^2/length(window);

stftAudio = stft(audio, audioInfo.SampleRate, ...
    "Window", window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange','onesided');

powerSpecAudio = abs(stftAudio).^2/length(window);

%% Mix noise into clean audio

%get random initial value from noise signal
randind = randi(numel(washNoise) - numel(audio), [1 1]);

%get segment from noise corresponding to length of clean audio signal
noiseSegment = washNoise(randind:randind + numel(audio) - 1);

speechPower = sum(audio.^2);
noisePower = sum(noiseSegment.^2);

noiseSegment = noiseSegment.*sqrt(speechPower/noisePower);

noisyAudio = audio + noiseSegment;

%% TODO: Load data into tall arrays and normalize... Maybe even employ sample rate conversion?

[audioMag, noisedMag] = transformSignal(audio, fs, washNoise, 0.015, dftSize*2);

%%
divisionFactor = 8;
dividedSegmentLength = segmentLength/divisionFactor;

%collect last bit of audio
audioSegment = powerSpecAudio(size(powerSpecAudio,2)-dividedSegmentLength:end);
rev_as = powerSpecAudio(1:dividedSegmentLength);

%concatenate the 
noisedMag = [noisedMag(:,1:divisionFactor-1) noisedMag];



%% Define network structure

%TODO: evaluate this very simple structure once data is ready...

 inputSize = [dftSize, 1, 1]; %should input size be of dftSize or the window?
 %numClasses = 2;
% 
layers = [
    imageInputLayer(inputSize)
    %convolution2dLayer(5,20)
    convolution2dLayer([1024 1],1,Stride=[1 100],Padding="same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1024)
    softmaxLayer
    %classificationLayer];
    regressionLayer];

analyzeNetwork(layers)

%% Define training parameters
%TODO: how do we measure the loss?

miniBatchSize = samplesIncluded*0.1;
options = trainingOptions("adam", ... %CHECK: why adam instead of regular gradient descend?
    MaxEpochs=3, ...
    InitialLearnRate=1e-5, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ... %ValidationFrequency=floor(size(trainPredictors,4)/miniBatchSize), ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.9, ...
    LearnRateDropPeriod=1 );
    %ValidationData={validatePredictors,permute(validateTargets,[3 1 2 4])});

    %%

    %liteConvNet = trainNetwork(?,layers,options)
