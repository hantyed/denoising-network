%for downloading whole commonvoice dataset (~1 GB)
downloadFolder = matlab.internal.examples.downloadSupportFile("audio","commonvoice.zip");
dataFolder = "./voiceData";
unzip(downloadFolder,dataFolder);