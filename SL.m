function Final_features=SL(Data,row,col,num_feature,StackFeature,layernum)
%% Self-expression layer of DIKS
    X_joint = [];
    for i=1:layernum
        X_joint = [X_joint StackFeature{i}.feature];
    end
    X_joint = [X_joint reshape(Data,row*col,num_feature)];
    X_joint_mean = mean(X_joint);
    X_joint_std = std(X_joint)+1;
    X_joint = bsxfun(@rdivide, bsxfun(@minus, X_joint, X_joint_mean), X_joint_std);
    [Final_features,W]=self_express2(X_joint);
    
    
 
end