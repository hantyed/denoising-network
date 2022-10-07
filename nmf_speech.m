%NMF dictionary of speech signals

clear, %close all;

%import data to matrices
%[input, fs] = audioread('speech_example.wav');
%[trainNoise, fsNoise] = audioread('mainsBrum50Hz.wav');
%[trainNoise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");
trainSpeech = getSpeechExamples;

%trainData = vertcat(trainSpeech, trainNoise);

%Train = cell2mat(trainSpeech.Files);

fs = 48000;

T = readall(trainSpeech);

trainData = cell2mat(T);

%trainData = T;
input = trainData;

% aud = zeros(size(T,1));
% fs = zeros(size(T,1));
% 
% for ii=1:size(T,1)
%     [aud(ii), fs(ii)] = read(trainSpeech);
% end

%specify number of sources
sources = 5;

%windowing conditions
segmentTime = 0.030;
segmentLength = round(segmentTime*fs);
overlap = round(segmentLength*0.5);
dftSize = round(4069*0.5);

window = sqrt(hann(segmentLength, 'periodic'));

%% compute STFT/magnitude of training data

stftTrain = stft(trainData, fs, ...
    'Window', window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange', 'onesided');

powerSpecT = abs(stftTrain).^2/length(window);

%% NMF

%initial dictionary A and activations X
A_train = rand(size(powerSpecT,1),sources);
X_train = rand(sources,size(powerSpecT,2));

iterations = 200;
beta = 1;

% Unsupervised training to obtain dictionary (atoms of piano notes)

[A_train, X_train, cost_T] = unsupervisedNmfTrain(powerSpecT, ...
    A_train, X_train, iterations, beta);

%% Supervised training to obtain activation matrix from recording (using known dictionary)

%compute STFT/magnitudes of input signal/recording

stftRec = stft(input, fs, ...
    'Window', window, 'OverlapLength', overlap, ...
    'FFTLength', dftSize, 'FrequencyRange', 'onesided');

powerSpecRec = abs(stftRec).^2/length(window);

X_rec = rand(sources,size(powerSpecRec,2));

[X_rec, cost_R] = supervisedNmfTrain(powerSpecRec, A_train, X_rec, iterations, beta);
    
%% seperate sources with Wiener filtering 

filteredSources = sourceSeperation(stftRec, A_train, X_rec, window, overlap);
    
%% Plot spectrograms

% f1 = figure;
% spectrogram(filteredSources(1,:),window,overlap,dftSize,'yaxis');
% 
% f2 = figure;
% spectrogram(filteredSources(2,:),window,overlap,dftSize,'yaxis');

%% NMF for identifying dictionary and activations (unsupervised training)

function [A, X, cost] = unsupervisedNmfTrain(S, A, X, iterations, beta)

    ap = A*X; %approximation of signal inputs S
    [F,N] = size(S);
    
    cost = zeros(1,iterations);
    cost(1) = betaDivergenceCost(S, ap, beta);

    for i=2:iterations

        A = A .* ((S .* ap.^(beta-2))* X') ./ (ap.^(beta-1) * X');
        ap = A*X; %recalculate approximate

        X = X .* (A'*(ap .^ (beta-2) .* S)) ./ (A'*ap.^ (beta-1));
        ap = A*X;

        %normalisation of matrices
        scale = sqrt(sum(A.^2,1));
        A = A .* repmat(scale.^-1,F,1);
        X = X .* repmat(scale',1,N);

        % Compute cost value
        cost(iterations) = betaDivergenceCost(S, ap ,beta);

    end   
end

%% Find activations 

function [Act, cost] = supervisedNmfTrain(S, dict, Act, iterations, beta)

    ap = dict*Act; %approximation of input signal
    
    cost = zeros(1,iterations);
    cost(1) = betaDivergenceCost(S, ap, beta);

    for i=2:iterations
        
        %update activation
        Act = Act .* (dict'*(ap .^ (beta-2) .* S)) ./ (dict'*ap.^ (beta-1));
        ap = dict*Act;
        
        %compute cost value
        cost(iterations) = betaDivergenceCost(S, ap, beta);
        
    end
end

%% Find cost of divergence depending on value of beta

function cost = betaDivergenceCost(S, S_ap, beta)

    if beta == 0 %IS
        cost = sum(S(:) ./ S_ap(:) - log10( S(:) ./ S_ap(:) ) - 1 );
        
    elseif beta == 1 % KL
        cost = sum(S(:) .* log( S(:) ./ S_ap(:) ) - S(:) + S_ap(:));

    elseif beta == 2 %Euclidean (squared error)
        cost = sum( 0.5 * (S(:) - S_ap(:)).^2 );
        
    end
end

%%

function sourceMatrix = sourceSeperation(stft, A, X, window, overlap)

    segmentLength = length(window);

    %number of sources and windowed segments found from X [K x N]
    [nSources, nSegments] = size(X);
    
    shiftedSamples = (segmentLength - overlap);
    sourceMatrix = zeros(nSources, nSegments * shiftedSamples + overlap);
    
    ap = A*X; %signal approximation
    
    for i = 1:nSources
        
        %Wiener reconstruction
        halfStft = stft .* (A(:,i) * X(i,:)) ./ ap;
        
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

%%

function trainDataVoices = getSpeechExamples

dataset = fullfile("voiceData","commonvoice"); 
trainDataVoices = audioDatastore(fullfile(dataset,"train"),"IncludeSubfolders",true);

%[noise, fsNoise] = audioread("WashingMachine-16-8-mono-1000secs.mp3");

samplesIncluded = 5;
trainDataVoices = shuffle(trainDataVoices);
trainDataVoices = subset(trainDataVoices,1:samplesIncluded); 

end
