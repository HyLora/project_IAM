function [normFeats] = normalize_featsTest(features,mn,st)

features = features';

normFeats =  (features - repmat(mn,size(features,1),1))./repmat(st,size(features,1),1);