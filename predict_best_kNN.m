function [predicted_label, rate] = predict_best_kNN(kNeighbors, bestModel, features2, label2)
    
% test using the best kNN model
predicted_label = predict(bestModel,features2);
    
% measure the performance
correct = 0;
for i=1:length(predicted_label)
    if predicted_label(i)==label2(i)
        correct=correct+1;            
    end
end
    
rate = (correct/length(predicted_label))*100;
disp(['recognition rate:', mat2str(rate)])


disp('----------results----------------')
disp(['the maximum recognition rate is ',mat2str(rate)])
disp(['and it is achieved with ',mat2str(kNeighbors),' nearest neighbors'])
disp(' ')