function [] = realize_confusion_matrix(predicted_label,ground_truth, title)

% using the corresponding function
figure
C = confusionmat(ground_truth, predicted_label);
cm = confusionchart(C, {'Alternative_Indie' 'Rock' 'Pop'}, 'Title', title, 'RowSummary', 'row-normalized'); 