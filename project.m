clear;clc
addpath(genpath(pwd))
windowLength = 0.03;
stepLength = 0.01;
nfft = 1024;

SNR = 5;

[chromaAltIndieTrain,mfccAltIndieTrain] = extract_from_path('alternative_indie/train/','*mp3', windowLength, stepLength);

[chromaAltIndieTest,mfccAltIndieTest] = extract_from_path('alternative_indie/test/','*mp3', windowLength, stepLength);

%chromaAltIndie = [chromaAltIndieTrain chromaAltIndieTest]; % add train and test

mfccAltIndie = [mfccAltIndieTrain mfccAltIndieTest];

[chromaPopTrain,mfccPopTrain]= extract_from_path('pop/train/','*mp3', windowLength, stepLength);

[chromaPopTest,mfccPopTest] = extract_from_path('pop/test/','*mp3', windowLength, stepLength);

chromaPop = [chromaPopTrain chromaPopTest]; % add train and test

%mfccPop = [mfccPopTrain mfccPopTest];

[chromaRockTrain,mfccRockTrain] = extract_from_path('rock/train/','*mp3', windowLength, stepLength);

[chromaRockTest,mfccRockTest] = extract_from_path('rock/test/','*mp3', windowLength, stepLength);

chromaRock = [chromaRockTrain chromaRockTest]; % add train and test

%mfccRock = [mfccRockTrain mfccRockTest];

%Concatenazione Chroma Train
[labelChromaAltIndie,labelChromaPop,labelChromaRock] = set_labels(chromaAltIndieTrain,chromaPopTrain,chromaRockTrain);
all_train_chroma = [chromaAltIndieTrain chromaPopTrain chromaRockTrain];
all_labels_chroma = [labelChromaAltIndie;labelChromaPop;labelChromaRock];

%Normalizzazione Chroma Train
[all_train_chroma,mn_train_chroma,st_train_chroma] = normalize_feats(all_train_chroma);

%Concatenazione Mfcc Train
[labelMfccAltIndie,labelMfccPop,labelMfccRock] = set_labels(mfccAltIndieTrain,mfccPopTrain,mfccRockTrain);
all_train_mfcc = [mfccAltIndieTrain mfccPopTrain mfccRockTrain];
all_labels_mfcc = [labelMfccAltIndie;labelMfccPop;labelMfccRock];

%Normalizzazione Mfcc Train
[all_train_mfcc,mn_all_mfcc,st_all_mfcc] = normalize_feats(all_train_mfcc);

% concatenating chromogram+mfcc Train

allFeatsAltIndieTrain = [chromaAltIndieTrain; mfccAltIndieTrain];

allFeatsPopTrain = [chromaPopTrain; mfccPopTrain];

allFeatsRockTrain = [chromaRockTrain; mfccRockTrain];

[labelallFeatsAltIndie,labelallFeatsPop,labelallFeatsRock] = set_labels(allFeatsAltIndieTrain,allFeatsPopTrain,allFeatsRockTrain);

allFeatsTrain = [allFeatsAltIndieTrain allFeatsPopTrain allFeatsRockTrain];

allFeatsLabels = [labelallFeatsAltIndie; labelallFeatsPop; labelallFeatsRock];

% normalize chromogram + mfcc
[allFeatsTrain,mn_feats_train,st_feats_train] = normalize_feats(allFeatsTrain);

%adding chroma Test
[labelChromaAltIndieTest,labelChromaPopTest,labelChromaRockTest] = set_labels(chromaAltIndieTest,chromaPopTest,chromaRockTest);

all_test_chroma = [chromaAltIndieTest chromaPopTest chromaRockTest];
ground_truth_chroma = [labelChromaAltIndieTest;labelChromaPopTest;labelChromaRockTest];

% normalize the test set
all_test_chroma = normalize_featsTest(all_test_chroma,mn_train_chroma,st_train_chroma);

% adding mfcc Test
[labelMfccAltIndieTest,labelMfccPopTest,labelMfccRockTest] = set_labels(mfccAltIndieTest,mfccPopTest,mfccRockTest);
all_test_mfcc = [mfccAltIndieTest mfccPopTest mfccRockTest];
ground_truth_mfcc = [labelMfccAltIndieTest;labelMfccPopTest;labelMfccRockTest];

% normalize the test set
all_test_mfcc = normalize_featsTest(all_test_mfcc,mn_all_mfcc,st_all_mfcc);

% adding chromogram and MFCCs test
allFeatsAltIndieTest = [chromaAltIndieTest; mfccAltIndieTest];
allFeatsPopTest = [chromaPopTest; mfccPopTest];
allFeatsRockTest = [chromaRockTest; mfccRockTest];

allFeatsTest = [allFeatsAltIndieTest allFeatsPopTest allFeatsRockTest];

[labelAllFeatsAltIndieTest,labelAllFeatsPopTest,labelAllFeatsRockTest] = set_labels(allFeatsAltIndieTest,allFeatsPopTest,allFeatsRockTest);

ground_truth_allFeats = [labelAllFeatsAltIndieTest; labelAllFeatsPopTest; labelAllFeatsRockTest]; % contains the real labels

[allFeatsTest] = normalize_featsTest(allFeatsTest,mn_feats_train,st_feats_train);

k = [1 10 20];

%KNN Chroma

disp('----------KNNChroma----------------')

%function [predicted_label, rate] = kNN(k, features1, label1, features2, label2,idx)

[plabel_chroma,rateChroma] = kNN(k,all_train_chroma,all_labels_chroma,all_test_chroma,ground_truth_chroma);

%KNN Mfcc
disp('----------KNNMfcc----------------')
[plabel_mfcc,rateMfcc] = kNN(k,all_train_mfcc,all_labels_mfcc,all_test_mfcc,ground_truth_mfcc);

%KNN All
disp('----------KNNAll----------------')
[plabel_All,rateAll,MdlKnnAll] = kNN(k,allFeatsTrain,allFeatsLabels,allFeatsTest,ground_truth_allFeats);

%adding bang noise to alternative_indie

add_noise('bang.mp3','alternative_indie/test/','*mp3',SNR,'alternative_indie_test_noise/test/')

%adding bang noise to pop

add_noise('bang.mp3','pop/test/','*mp3',SNR,'pop_test_noise/test/')

%adding bang noise to rock

add_noise('bang.mp3','rock/test/','*mp3',SNR,'rock_test_noise/test/')


%KNN bang All

[chromaAltIndieBangTest,mfccAltIndieBangTest] = extract_from_path('alternative_indie_test_noise/test/','*wav', windowLength, stepLength);
[chromaPopBangTest,mfccPopBangTest] = extract_from_path('pop_test_noise/test/','*wav', windowLength, stepLength);
[chromaRockBangTest,mfccRockBangTest] = extract_from_path('rock_test_noise/test/','*wav', windowLength, stepLength);


% adding chromogram+mfcc Test

allFeatsAltIndieBangTest = [chromaAltIndieBangTest; mfccAltIndieBangTest];

allFeatsPopBangTest = [chromaPopBangTest; mfccPopBangTest];

allFeatsRockBangTest = [chromaRockBangTest; mfccRockBangTest];

[labelallFeatsAltIndieBang,labelallFeatsPopBang,labelallFeatsRockBang] = set_labels(allFeatsAltIndieBangTest,allFeatsPopBangTest,allFeatsRockBangTest);

allFeatsTestBang= [allFeatsAltIndieBangTest allFeatsPopBangTest allFeatsRockBangTest];


ground_truth_allFeatsBang = [labelallFeatsAltIndieBang; labelallFeatsPopBang; labelallFeatsRockBang];

% normalize chromogram+mfcc

allFeatsTestBang = normalize_featsTest(allFeatsTestBang,mn_feats_train,st_feats_train);

%KNN Bang All
disp('----------KNNBang----------------')
[predLAllBang, rateAllBang] = predict_best_kNN(10,MdlKnnAll,allFeatsTestBang,ground_truth_allFeatsBang);

%% --- SPECTRAL SUBSTRACTION / KNN5 ---
% --- sub noise ---
disp('.......... denoising the test-noisyFiles and saving in the new enhanced-paths ..........')

sub_noise('alternative_indie_test_noise/test/', '*wav', 'altIndie_enh_test/') % denoising and saving denoised alt in the new enhanced-path
sub_noise('pop_test_noise/test/', '*wav', 'pop_enh_test/') % denoising and saving denoised alt in the new enhanced-path
sub_noise('rock_test_noise/Test/', '*wav', 'rock_enh_test/') % denoising and saving denoised alt in the new enhanced-path

% --- MFCCs kNN 5 (applied to enhanced-files, in the enhanced-paths) ---
% extract mfccs from directories' enhanced-paths 
disp('extracting features from enhanced files...')
disp(' ')

%KNN Bang All

[chromaAltIndieEnhTest,mfccAltIndieEnhTest] = extract_from_path('altIndie_enh_test/','*wav', windowLength, stepLength);
[chromaPopEnhTest,mfccPopEnhTest] = extract_from_path('pop_enh_test/','*wav', windowLength, stepLength);
[chromaRockEnhTest,mfccRockEnhTest] = extract_from_path('rock_enh_test/','*wav', windowLength, stepLength);


% adding chromogram+mfcc Test
allFeatsAltIndieEnhTest = [chromaAltIndieEnhTest; mfccAltIndieEnhTest];

allFeatsPopEnhTest = [chromaPopEnhTest; mfccPopEnhTest];

allFeatsRockEnhTest = [chromaRockEnhTest; mfccRockEnhTest];

[labelallFeatsAltIndieEnh,labelallFeatsPopEnh,labelallFeatsRockEnh] = set_labels(allFeatsAltIndieEnhTest,allFeatsPopEnhTest,allFeatsRockEnhTest);

allFeatsTestEnh = [allFeatsAltIndieEnhTest allFeatsPopEnhTest allFeatsRockEnhTest];


ground_truth_allFeatsEnh = [labelallFeatsAltIndieEnh; labelallFeatsPopEnh; labelallFeatsRockEnh];

% normalize chromogram+mfcc
allFeatsTestEnh = normalize_featsTest(allFeatsTestEnh,mn_feats_train,st_feats_train);


%KNN Enh All
disp('----------KNNEnh----------------')
[predlAllEnh, ratelAllEnh]=predict_best_kNN(10,MdlKnnAll,allFeatsTestEnh,ground_truth_allFeatsEnh);


%confusion matrix chroma
confmatrix(plabel_chroma,ground_truth_chroma)


%confusion matrix mfcc
confmatrix(plabel_mfcc,ground_truth_mfcc)


%confusion matrix all
confmatrix(plabel_All,ground_truth_allFeats)


%confusion matrix noise (ground_truth_allFeatsBang)
confmatrix(predLAllBang,ground_truth_allFeatsBang)


%confusion matrix enh
confmatrix(predlAllEnh,ground_truth_allFeatsEnh)


%% --- CONFUSION MATRIX ---
disp('.......... realizing confusion matrixes ..........')
disp(' ')

% --- chromagram ---
disp('1st confusion matrix: chromagram ...')
realize_confusion_matrix(plabel_chroma,ground_truth_chroma, 'Confusion matrix: CHROMA') % function for confusion matrix

% --- mfccs ---
disp('2nd confusion matrix: mfccs ...')
realize_confusion_matrix(plabel_mfcc,ground_truth_mfcc, 'Confusion matrix: MFCCs') % function for confusion matrix

% --- chromagram + mfccs ---
disp('3rd confusion matrix: chromagram + mfccs ...')
realize_confusion_matrix(plabel_All,ground_truth_allFeats, 'Confusion matrix: CHROMA + MFCCs') % function for confusion matrix

% --- mfccs noisy ---
disp('4th confusion matrix: All noisy ...')
realize_confusion_matrix(predLAllBang,ground_truth_allFeatsBang, 'Confusion matrix: All noisy') % function for confusion matrix

% --- mfccs enhanced ---
disp('5th confusion matrix: All enhanced ...')
realize_confusion_matrix(predlAllEnh,ground_truth_allFeatsEnh, 'Confusion matrix: All enhanced') % function for confusion matrix
