function [chroma,mfccs] = extract_from_path(path,extension,windowLength,stepLength)
addpath(genpath(path))
files = dir([path,'*.',extension]);
chroma=[];
mfccs = [];
for i=1:length(files)
    Chroma=chroma_features(files(i).name,windowLength,stepLength);	
    chroma = [chroma Chroma];
    ceps = extractMFCCs(files(i).name,windowLength,stepLength);
    mfccs = [mfccs ceps];
end