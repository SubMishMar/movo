clear all
close
clc
M = importdata('00.txt');

for i=1:1
    T = reshape(M(i,:), [4,3])';
    disp(T)
    disp(M(i,:))
end