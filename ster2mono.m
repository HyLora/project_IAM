function [yMono,fs] = ster2mono(filename)
[y,fs] = audioread(filename);
if size(y,2) == 2 % yes it is stereo
    yMono = sum(y,2)/2;
    
    
else
    yMono=y;
    
end
    