function [labe_patch,ww,win_inter]=CSL(Data,row, col, num_feature,w,K,P)
%Creat segmentation PATCH
%%Data is the input HSI.
%%Y1 is base image obtained by pca, P is the number of superpixels
Y1=reshape(Data,row*col,num_feature);
Y1=normalize(Y1);
B=pca(Y1);
Y1=Y1*B;
Y1=reshape(Y1,row,col);
[Superpixel_labels,bmapOnImg]=suppixel(Y1,P);
[labe_patch,ww]=align_padding(Superpixel_labels,K,w,P);
win_inter = (ww-1)/2;