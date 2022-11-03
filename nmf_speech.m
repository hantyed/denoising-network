%NMF dictionary of speech signals

clear, %close all;
doSampleRateConvert = false; %whether or not the audio is downsampled
unknownInput = false; %whether to use new data after dictionary is created

%import data to matrices
%[input, fs] = audioread('speech_example.wav');
%[trainNoise, fsNoise] = audioread('./nmfData/mainsBrum50Hz.wav');
[trainNoise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
%[trainNoise, fsNoise] = audioread("WashingMachine_48k_120secs.wav");

samplesIncluded = 5;
trainSpeech = getSpeechSamples(samplesIncluded);

%get all samples from data storage and convert to a vector
T = readall(trainSpeech);
audioData = cell2mat(T);

%convert the sample rate of the speech signals if needed
if doSampleRateConvert

    inputFs = 48000;
    fs = 8000;
    bandwidth = 7920; %CHECK: why this value?
    
    sampleRateConv = dsp.SampleRateConverter('InputSampleRate', inputFs,...
        'OutputSampleRate', fs, 'Bandwidth', bandwidth);

    audioData = sampleRateConv(audioData);

else

    fs = 48000;

end

%convert the sample rate of the noise signal to be the same as audioData
if fsNoise ~= fs
 
    inputFs = fsNoise;
    bandwidth = 7920;
    decimationFactor = inputFs/fs;
    L = floor(numel(trainNoise)/decimationFactor);
    trainNoise = trainNoise(1:decimationFactor*L);

    sampleRateConv = dsp.SampleRateConverter('InputSampleRate', inputFs,...
        'OutputSampleRate', fs, 'Bandwidth', bandwidth);

    trainNoise = sampleRateConv(trainNoise);
end

%%

%add noise to the signals - either using a new set of samples or trainData

if unknownInput
    inputInit = getSpeechSamples(samplesIncluded);
    inputInit = readall(inputInit);
    inputInit = cell2mat(inputInit);
    
    input = add_noise(inputInit, trainNoise);
else

    input = add_noise(audioData, trainNoise);
end

trainData = vertcat(audioData,trainNoise(1:fs*2));

%% set data parameters

%specify number of sources
sources = 5;

%windowing conditions
segmentTime = 0.030;
segmentLength = round(segmentTime*fs);
<<<<<<< Updated upstream
overlap = round(segmentLength*0.75);
dftSize = round(4069*2);
=======
overlap = round(segmentLength*.75);
dftSize = round(4069*0.5);
>>>>>>> Stashed changes

window = sqrt(hann(segmentLength, 'periodic'));

%% compute STFT/magnitude of training data

stftTrain = stft(trainData, fs, ...
    'Window', window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange', 'onesided');

powerSpecT = abs(stftTrain).^2/length(window);

% figure;
% plot(powerSpecT)


%% NMF
%TODO: figure out WHY output sometimes turns out NAN

%initial dictionary A and activations X
A_train = rand(size(powerSpecT,1),sources);
X_train = rand(sources,size(powerSpecT,2));

iterations = 200;
<<<<<<< Updated upstream
beta = 2;
=======
beta = 1;
%%
>>>>>>> Stashed changes

% Unsupervised training to obtain dictionary

[A_train, X_train, cost_T] = unsupervisedNmfTrain(powerSpecT, ...
    A_train, X_train, iterations, beta);

<<<<<<< Updated upstream
=======

%%
figure;
for i=1:sources
plot(X_train(i,:))
hold on
end
hold off

%% Supervised training to obtain activation matrix from recording (using known dictionary)
>>>>>>> Stashed changes

%%
%plotX(X_train, sources)
%plotX(activationsSummed, sources)

%% Supervised training to obtain activation matrix from recording (using known dictionary)
 
%compute STFT/magnitudes of input signal/recording
stftInput = stft(input, fs, ...
    'Window', window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange', 'onesided');

powerSpecIn = abs(stftInput).^2/length(window);

X_input = rand(sources,size(powerSpecIn,2));

[X_input, cost_R] = supervisedNmfTrain(powerSpecIn, ...
    A_train, X_input, iterations, beta);

%% separate sources with Wiener filtering

filteredSources = sourceSeperation(stftInput, A_train, X_input, window, overlap);


%% plot spectrogram of each source

plotSpec(filteredSources,sources,window,overlap,dftSize)

%% combine wanted sources in one signal

x = 1;
nosySources=[3 4]; %I estimate this from looking at the spectrograms each time...

for ii=1:sources
    if ii ~= nosySources
        x = x + filteredSources(ii,:);
    end
end
%sound(x,fs)

%% NMF for identifying dictionary and activations (unsupervised training)

function [A, X, cost] = unsupervisedNmfTrain(powerSpectrum, A, X, iterations, beta)

    ap = A*X; %approximation of signal inputs S
    [F,N] = size(powerSpectrum);
    
    cost = zeros(1,iterations);
    cost(1) = betaDivergenceCost(powerSpectrum, ap, beta);

    for i=2:iterations

        A = A .* ((powerSpectrum .* ap.^(beta-2))* X') ./ (ap.^(beta-1) * X');
        ap = A*X; %recalculate approximate

        X = X .* (A'*(ap .^ (beta-2) .* powerSpectrum)) ./ (A'*ap.^ (beta-1));
        ap = A*X;

        %normalisation of matrices
        scale = sqrt(sum(A.^2,1));
        A = A .* repmat(scale.^-1,F,1);
        X = X .* repmat(scale',1,N);

        % Compute cost value
        cost(iterations) = betaDivergenceCost(powerSpectrum, ap ,beta);

    end   
end

%% Find activations 

function [X, cost] = supervisedNmfTrain(powerSpectrum, A, X, iterations, beta)

    ap = A*X; %approximation of input signal
    
    cost = zeros(1,iterations);
    cost(1) = betaDivergenceCost(powerSpectrum, ap, beta);

    for i=2:iterations
        
        %update activation
        X = X .* (A'*(ap .^ (beta-2) .* powerSpectrum)) ./ (A'*ap.^ (beta-1));
        ap = A*X;
        
        %compute cost value
        cost(iterations) = betaDivergenceCost(powerSpectrum, ap, beta);
        
    end
end

%% Find cost of divergence depending on value of beta

function cost = betaDivergenceCost(powerSpectrum, S_ap, beta)

    if beta == 0 %IS
        cost = sum(powerSpectrum(:) ./ S_ap(:) - log10( powerSpectrum(:) ...
            ./ S_ap(:) ) - 1 );
        
    elseif beta == 1 % KL
        cost = sum(powerSpectrum(:) .* log( powerSpectrum(:) ...
            ./ S_ap(:) ) - powerSpectrum(:) + S_ap(:));

    elseif beta == 2 %Euclidean (squared error)
        cost = sum( 0.5 * (powerSpectrum(:) - S_ap(:)).^2 );
        
    end
end

%% Seperation with wiener filter

function sourceMatrix = sourceSeperation(stft, A, X, window, overlap)

    segmentLength = length(window);

    %number of sources and windowed segments found from X [K x N]
    [nSources, nSegments] = size(X);
    
    shiftedSamples = (segmentLength - overlap);
    sourceMatrix = zeros(nSources, nSegments * shiftedSamples + overlap);
    
    ap = A*X; %signal approximation
    
    for i = 1:nSources
        
        %Wiener reconstruction
        halfStft = stft .* (A(:,i) * X(i,:)) ./ dynamicRangeLimiting(ap,120);
        
        %create dublicate freq components
        fullStft = [halfStft; flipud(conj(halfStft(2:end-1,:)))];
        
        %revert signal to time domain
        oversamplesSeparatedSource = ifft(fullStft);
        separatedSourceMatrix = real(oversamplesSeparatedSource(1:segmentLength,:));
        
        index = 1:segmentLength;
        
        for s = 1:nSegments
            
            sourceMatrix(i,index) = sourceMatrix(i,index) + ...
                (window.*separatedSourceMatrix(:,s))';
            
            index = index + shiftedSamples;
        end
        
    end
    
end

%% dynamicRangeLimiting from SMSA to diminish the amplitude (to be modified?)

function limitedNonnegativeData = ...
    dynamicRangeLimiting(nonnegativeData, maxRangeDb)

    logPowerSpectrum = 10*log10(nonnegativeData);

    limitedLogPowerSpectrum = max(logPowerSpectrum, ...
            max(max(logPowerSpectrum) - maxRangeDb));

    limitedNonnegativeData = 10.^(limitedLogPowerSpectrum/10);
end

%% load data

function trainDataVoices = getSpeechSamples(samplesIncluded)

    dataset = fullfile("voiceData","commonvoice"); 
    trainDataVoices = audioDatastore(fullfile(dataset,"train"),"IncludeSubfolders",true);
    
    %[noise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
    
    trainDataVoices = shuffle(trainDataVoices);
    trainDataVoices = subset(trainDataVoices,1:samplesIncluded); 

end

%% plottig helpers
function plotX(X,times)

    figure,
    
    for ii=1:times
    
        hold on, plot(X(ii,:))
    end
end

function plotSpec(X,times,window,overlap,dftSize)

    for ii =1:times

        figure, spectrogram(X(ii,:),window,overlap,dftSize,'yaxis');
    end
end

%% "Cheat" - Remove the activations that are not of interest 
%(if we know when different sources are active in which sections of the
%training data, i.e. 


function [activations, activationsSummed] = pickActivationsBySections(X, T, samplesIncluded)
% X = activation matrix, T = data cell

    activations = zeros(size(X));
    activationsSummed = zeros(size(X));
    prevFrames = 1;
    
    for n = 1:samplesIncluded
        %rounding errors might occur for these values since they need to be
        %integers... condition check beforehand might help
        sampleSize = round(size(T{n},1));
        procentage = round(sampleSize / size(input,1) * 100);
    
        sampleTimeFrame = round((size(X,2) * procentage)/100);
    
        %activations for each entry in X (assuming they match time instances consecutively)
        activations(n, prevFrames:prevFrames+sampleTimeFrame) = ...
            X(n, prevFrames:prevFrames+sampleTimeFrame);
    
        %combined activations across all dictionary entries for each time instance
        for s = 1:samplesIncluded  
            activationsSummed(s, prevFrames:prevFrames+sampleTimeFrame) = ...
                activationsSummed(s, prevFrames:prevFrames+sampleTimeFrame) + ...
                X(s, prevFrames:prevFrames+sampleTimeFrame);
        end
    
        prevFrames = prevFrames + sampleTimeFrame;
    
    end
end