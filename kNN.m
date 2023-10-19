function [predicted_label, rate,Mdl] = kNN(k, features1, label1, features2, label2)

rate = [];
for kk=1:length(k)
    disp(['set-up the kNN... number of neighbors: ',mat2str(k(kk))])
    Mdl = fitcknn(features1,label1','NumNeighbors',k(kk));
    
    % test the kNN
    predicted_label = predict(Mdl,features2);
    
    % measure the performance
    correct = 0;
    for i=1:length(predicted_label)
        if predicted_label(i)==label2(i)
            correct=correct+1;            
        end
    end
    
    rate(kk) = (correct/length(predicted_label))*100;
    disp(['recognition rate: ', mat2str(rate(kk))])
end
[a,b]=max(rate);

disp('----------results----------------')
disp(['the maximum recognition rate is ',mat2str(a)])
disp(['and it is acheived with ',mat2str(k(b)),' nearest neighbors'])
% train using the best k

Mdl = fitcknn(features1,label1','NumNeighbors',(k(b)));
predicted_label = predict(Mdl,features2);