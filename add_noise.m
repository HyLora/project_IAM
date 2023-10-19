function [] = add_noise(noise_file, path_clean_file, extension, SNR, saving_path)
    addpath(genpath(path_clean_file))
    cleanFiles = dir([path_clean_file, '*.', extension]);
    
    for i = 1:length(cleanFiles)
        disp(['adding noise ', noise_file, ' to file ', cleanFiles(i).name, ' at SNR ', mat2str(SNR), '...'])
        
        [y1Mono, fs] = ster2mono(cleanFiles(i).name);
        [y2Mono] = ster2mono(noise_file);
        
        % Repeat the noise signal to match the length of the clean audio
        y2Mono = repmat(y2Mono, ceil(length(y1Mono) / length(y2Mono)), 1);
        y2Mono = y2Mono(1:length(y1Mono));
        
        noisy_sig = sigmerge(y1Mono, y2Mono, SNR);
        
        disp('saving the noisy file...')
        noisyFile = ['noisy', cleanFiles(i).name(1:end-4), '_SNR', mat2str(SNR), '_noise', noise_file(1:end-4), '.wav'];
        audiowrite([saving_path, '\', noisyFile], noisy_sig, fs)
    end
end
