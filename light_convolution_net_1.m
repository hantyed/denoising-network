clear,
%prepare data and build a small convolution model

doSampleRateConvertNoise = true;
convStructure = 2; %1 or 2

%load noise sample and all data from commonvoice dataset
[noise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
dataset = fullfile("voiceData","commonvoice"); 

%% Load a selection of data samples 

trainDataVoices = audioDatastore(fullfile(dataset,"train"),"IncludeSubfolders",true);

samplesIncluded = 50; %load only a section of the dataset
trainDataVoices = shuffle(trainDataVoices);
trainDataVoices = subset(trainDataVoices,1:samplesIncluded);
fs = 48000; %to be extracted from data? -in case we use another dataset

%% Sample rate conversion for noise signal
%convert the sample rate of the noise signal to be the same as audioData
if fsNoise ~= fs && doSampleRateConvertNoise
 
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

%set values for data to be segmented
segments = 8;
window = sqrt(hann(1024,'periodic'));
overlap = round(length(window)*0.5);

dftSize = 1024; %TODO: figure out if this can be set to not depend on window size

%NB: tall is a keyword that by default starts a parallel pool
%when using a small data amount it runs faster with this setting deactivated
T = tall(audioData); 

[targets, predictors, magnitudes] = cellfun(...
    @(x)prepare_data(x,...
        noise,...
        window,...
        overlap,...
        dftSize ...
        ), ...
    T, ...
    UniformOutput=false ...
); 

[targets, predictors, magnitudes] = gather(targets, predictors, magnitudes);

%% Normalisation of data

predictors = cat(3,predictors{:});
noisyMean = mean(predictors(:));
noisyStd = std(predictors(:));
predictors(:) = (predictors(:) - noisyMean)/noisyStd;

targets = cat(2,targets{:});
cleanMean = mean(targets(:));
cleanStd = std(targets(:));
targets(:) = (targets(:) - cleanMean)/cleanStd;

%reshape
predictors = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targets = reshape(targets,1,1,size(targets,1),size(targets,2));

%% split into training and validation

features = dftSize*0.5;

inds = randperm(size(targets,4));
L = floor(0.90*size(targets,4));

trainPredictors = predictors(1:features,:,:,inds(1:L));
trainTargets = targets(:,:,1:features,inds(1:L));

validatePredictors = predictors(1:features,:,:,inds(L+1:end));
validateTargets = targets(:,:,1:features,inds(L+1:end));

%% Define network structure

if convStructure == 1

    %convolution model only using full feature size

%     layers = [
% 
%         imageInputLayer([features segments])
%     
%         repmat([...
%         convolution2dLayer([5 1],30,Stride=[1 100],Padding="same")
%         convolution2dLayer([5 1],30,Stride=[1 100],Padding="same")
%         convolution2dLayer([5 1],18,Stride=[1 100],Padding="same")
%         batchNormalizationLayer
%         reluLayer ...
%         maxPooling2dLayer([2 2],Stride=[2 2])...
%         ],4,1);
%     
%     
%         convolution2dLayer([5 1],30,Stride=[1 100],Padding="same")
%         batchNormalizationLayer
%         reluLayer
%     
%         convolution2dLayer([5 1],8,Stride=[1 100],Padding="same")
%         batchNormalizationLayer
%         reluLayer
%     
%         denoise example uses feature size for the last conv layer... 
%         presumably to make a full connection before regression...
%         convolution2dLayer([features 1],1,Stride=[1 100],Padding="same")
%     
%         This is the output layer, which will measure the loss as MSE
%         regressionLayer
    %];
end


if convStructure == 2

%small autoencoder with 2x downsampling

    layers = [

        imageInputLayer([512 8 1],"Name","imageinput")
    
        convolution2dLayer([9 8],18,Name="en_conv1", Stride=[1 100], Padding="same")
        batchNormalizationLayer(Name="en_batchnorm1")
        reluLayer(Name="en_relu1")
        maxPooling2dLayer([2 2],Name="en_maxpool1",Stride=[2 2], Padding="same")
    
        convolution2dLayer([5 1],30,Name="en_conv2",Stride=[1 100], Padding="same")
        batchNormalizationLayer(Name="en_batchnorm2")
        reluLayer(Name="en_relu2")
        maxPooling2dLayer([2 2],Name="en_maxpool2",Stride=[2 2], Padding="same")
    
%         convolution2dLayer([5 1],30,Name="en_conv3",Stride=[1 100], Padding="same")
%         batchNormalizationLayer(Name="en_batchnorm3")
%         reluLayer(Name="en_relu3")
%         maxPooling2dLayer([2 2],Name="en_maxpool3",Stride=[2 2], Padding="same")
%     
%         transposedConv2dLayer([2 1],30,Name="de_transposed-conv3",Stride=[2 1], Cropping="same")
%         convolution2dLayer([5 1],30,Name="de_conv3", Stride=[1 100], Padding="same")
%         batchNormalizationLayer(Name="de_batchnorm3")
%         reluLayer(Name="de_relu3")

        transposedConv2dLayer([2 1],30,Name="de_transposed-conv1",Stride=[2 1], Cropping="same")
        convolution2dLayer([5 1],30,Name="de_conv1", Stride=[1 100], Padding="same")
        batchNormalizationLayer(Name="de_batchnorm1")
        reluLayer(Name="de_relu1")
    
        transposedConv2dLayer([2 2],18,Name="de_transposed-conv2",Stride=[2 2], Cropping="same")
        convolution2dLayer([9 1],18,Name="de_conv2", Stride=[1 100], Padding="same")
        batchNormalizationLayer(Name="de_batchnorm2")
        reluLayer("Name","de_relu2")
    
        convolution2dLayer([512 1],1,Name="conv5",Stride=[1 100], Padding="same")
        regressionLayer(Name="regressionoutput")
        ];
end

%% Define training parameters
%TODO: meaning behind these parameters? What's optimal?

miniBatchSize = floor(samplesIncluded*0.30);
options = trainingOptions("adam", ... %CHECK: why adam instead of regular gradient descend?
    MaxEpochs=3, ...
    InitialLearnRate=1e-5, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ValidationFrequency=floor(size(trainPredictors,4)/miniBatchSize), ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.9, ... 
    LearnRateDropPeriod=1, ...
    ValidationData={validatePredictors,permute(validateTargets,[3 1 2 4])});

    %% Train the network

    convNet = trainNetwork(...
        trainPredictors, ...
        permute(trainTargets,[3 1 2 4]), ...
        layers, ...
        options ...
    );
