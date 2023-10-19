function [normFeats,mn,st] = normalize_feats(features)
features = features';
mn = mean(features);
st = std(features);
normFeats =  (features - repmat(mn,size(features,1),1))./repmat(st,size(features,1),1);