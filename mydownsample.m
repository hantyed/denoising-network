function [input] = mydownsample(input, inputFs, targetFs) 

bandwidth = 7920;
decimationFactor = inputFs/targetFs;
L = floor(numel(input)/decimationFactor);
input = input(1:decimationFactor*L);

sampleRateConv = dsp.SampleRateConverter('InputSampleRate', inputFs,...
    'OutputSampleRate', targetFs, 'Bandwidth', bandwidth);

input = sampleRateConv(input);