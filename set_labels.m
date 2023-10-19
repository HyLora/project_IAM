function [label1, label2, label3] = set_labels(feats1, feats2, feats3)


label1 = repmat(1,length(feats1),1);
label2 = repmat(2,length(feats2),1);
label3 = repmat(3,length(feats3),1);


%{

[labelAllFeatsJazzTest,labelAllFeatsPopTest,labelAllFeatsRockTest] = 
set_labels(allFeatsJazzTest,allFeatsPopTest,allFeatsRockTest);

%}