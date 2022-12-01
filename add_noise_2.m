function [audio] = add_noise_2(audio)

channels = 1;

noisegenerator = dsp.ColoredNoise('blue', length(audio), channels, OutputDataType='double');
blueNoise = noisegenerator();

speechPower = sum(audio.^2);
noisePower = sum(blueNoise.^2);

noiseSegment = blueNoise.*sqrt(speechPower/noisePower);

audio = audio + noiseSegment;