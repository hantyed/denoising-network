function noisyAudio = add_noise(audio, noise)
% Mix noise into clean audio signal 

%extend noise signal if the audio signal is longer
if length(audio) > length(noise)
    vertcat(noise, noise);
end

%get random initial value from noise within a range that fits entire audio
randind = randi(numel(noise) - numel(audio), [1 1]);

%get segment from noise corresponding to length of clean audio signal
noiseSegment = noise(randind:randind + numel(audio) - 1);

speechPower = sum(audio.^2);
noisePower = sum(noiseSegment.^2);

noiseSegment = noiseSegment.*sqrt(speechPower/noisePower);

noisyAudio = audio + noiseSegment;

