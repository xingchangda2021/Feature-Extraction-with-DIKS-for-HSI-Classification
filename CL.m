function StackFeature=CL(Data,row,col,num_feature,Layernum,num_PC,P,labe_patch,w,win_inter)
%%Convolution layers of DIKS.

% w=29;
% win_inter = (w-1)/2;
epsilon = 0.01;


StackFeature= cell(Layernum,1);

for l=1:Layernum
    
    randidx = randperm(row*col);
    StackFeature{l}.centroids = zeros(w*w*num_PC,P);
    disp(['Extracting the features of the ',num2str(l),'th layer...']);
    if l==1
        
        XPCA = PCANorm(reshape(Data, row * col, num_feature),num_PC);
        
        XPCAvector = XPCA;
        minZ = min(XPCAvector);
        maxZ = max(XPCAvector);
        XPCAvector = bsxfun(@minus, XPCAvector, minZ);
        XPCAvector = bsxfun(@rdivide, XPCAvector, maxZ-minZ);
        
        
        XPCA_cov = cov(XPCA);
        [U S V] = svd(XPCA_cov);
        whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
        XPCA = XPCA * whiten_matrix;
        XPCA = bsxfun(@rdivide,bsxfun(@minus,XPCA,mean(XPCA,1)),std(XPCA,0,1)+epsilon);
        XPCA = reshape(XPCA,row,col,num_PC);
        X_extension = MirrowCut(XPCA,win_inter);
        
        for i=1:P
            index_col = ceil(randidx(i)/row);
            index_row = randidx(i) - (index_col-1) * row;
            %tem = X_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
             tem=X_extension(labe_patch{i}.a(1,1):labe_patch{i}.a(1,2),labe_patch{i}.a(2,1):labe_patch{i}.a(2,2),:);
%              tem=zero_sup(labe_patch{i}.p,tem);
            StackFeature{l}.centroids(:,i) = tem(:);
        end
        
        StackFeature{l}.feature = extract_features(X_extension,StackFeature{l}.centroids);
        
        XPCAvector = PCANorm([StackFeature{l}.feature],num_PC);
        minZ = min(XPCAvector);
        maxZ = max(XPCAvector);
        XPCAvector = bsxfun(@minus, XPCAvector, minZ);
        XPCAvector = bsxfun(@rdivide, XPCAvector, maxZ-minZ);
        
        clear StackFeature{l}.centroids;
    else
        XPCA = PCANorm(StackFeature{l-1}.feature,num_PC);
        
        XPCA_cov = cov(XPCA);
        [U S V] = svd(XPCA_cov);
        whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
        
        
        XPCA = XPCA * whiten_matrix;
        XPCA = bsxfun(@rdivide,bsxfun(@minus,XPCA,mean(XPCA,1)),std(XPCA,0,1)+epsilon);
        
        XPCA = reshape(XPCA,row,col,num_PC);
        X_extension = MirrowCut(XPCA,win_inter);
        
        for i=1:P
            index_col = ceil(randidx(i)/row);
            index_row = randidx(i) - (index_col-1) * row;
%             tem = X_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
             tem=X_extension(labe_patch{i}.a(1,1):labe_patch{i}.a(1,2),labe_patch{i}.a(2,1):labe_patch{i}.a(2,2),:);
            StackFeature{l}.centroids(:,i) = tem(:);
        end
        
        StackFeature{l}.feature = extract_features(X_extension,StackFeature{l}.centroids);
        
        XPCAvector = PCANorm(StackFeature{l}.feature,num_PC);
        minZ = min(XPCAvector);
        maxZ = max(XPCAvector);
        XPCAvector = bsxfun(@minus, XPCAvector, minZ);
        XPCAvector = bsxfun(@rdivide, XPCAvector, maxZ-minZ);
        
        clear StackFeature{l}.centroids;
    end
    
    clear X_extension;
    
end