function [] = sub_noise(pathNoise, extension, newPathEnh)

addpath(genpath(pathNoise))
fileList = dir([pathNoise, '*.',extension]);

for i=1:length(fileList)
    
    %[x1,fs]=audioread(fileList(i).name); % read noisy files
    %[x2,~]=audioread(babbleFilename); % read noise babble.wav
            
    noisyFile=[fileList(i).name]; % catching filenames of noisy audios
            
    enhFile=['enhanced',fileList(i).name(6:end)]; % output filename wanted for enhanced files, passed to specsub
            
    disp(['denoising ', fileList(i).name])
    disp('saving the enhanced file...')
    specsub(noisyFile,enhFile,newPathEnh) % denoises and creates an output file: enhFile (see function specsub)
            
end

disp(' ')