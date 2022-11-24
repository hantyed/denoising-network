dataFolder = tempdir;

datasetTrain = fullfile(dataFolder,"LibriSpeech","train-clean-100");
%if ~datasetExists(datasetTrain)
    filename = "train-clean-100.tar.gz";
	url = "http://www.openSLR.org/resources/12/" + filename;
    gunzip(url,dataFolder);
    unzippedFile = fullfile(dataFolder,filename);
    untar(unzippedFile{1}(1:end-3),dataFolder);
%end

%%

datasetDev = fullfile(dataFolder,"LibriSpeech","dev-clean");
if ~datasetExists(datasetDev)
    filename = "dev-clean.tar.gz";
	url = "http://www.openSLR.org/resources/12/" + filename;
    gunzip(url,dataFolder);
    unzippedFile = fullfile(dataFolder,filename);
    untar(unzippedFile{1}(1:end-3),dataFolder);
end

datasetTest = fullfile(dataFolder,"LibriSpeech","test-clean");
if ~datasetExists(datasetTest)
    filename = "test-clean.tar.gz";
	url = "http://www.openSLR.org/resources/12/" + filename;
    gunzip(url,dataFolder);
    unzippedFile = fullfile(dataFolder,filename);
    untar(unzippedFile{1}(1:end-3),dataFolder);
end