% Rough estimate of the size of data
all_data = nan(9000,896);
all_trialinfo = nan(1,9000);
indices = nan(1,21);
% For ease of calculation
indices(1) = 0;

% Iterate over all subjects
for j = 1:20
%     Load data file
    file_name = sprintf('data/subdata%d.mat',j);
    load(file_name);
    
%     Average and Normalize values to obtain feature vectors
    rows = size(datasave,1);
    no_feats = size(datasave,2)/50;
    no_samples = size(datasave,3);
    feature_values = zeros(no_feats,rows,no_samples);
    for row_no=1:no_samples
        unnorm_feats = zeros(no_feats,rows);    %     Swap rows and no_feats in unnorm_feats and feature_values for 14x64 and 64x14
        for i=1:no_feats
%             Average data into time windows of 50 ms each
            time_window = datasave(:,(i*50)-49:(i*50),row_no);
            avg_values = mean(time_window,2);
            unnorm_feats(i,:)=avg_values(:);
        end
%         Normalization
        min_feat = min(min(unnorm_feats));
        max_feat = max(max(unnorm_feats));
        feature_values(:,:,row_no) = (unnorm_feats - min_feat)/(max_feat - min_feat); 
    end

%     Put feature vectors into correct format
    features = reshape(feature_values,896,size(feature_values,3))';

%     Concatenate data
    if j>=2
        indices(j) = prev_end+indices(j-1);
    end
    curr_end = size(features,1);
    all_data(indices(j)+1:indices(j)+curr_end, :) = features;
    prev_end = curr_end;

%     Load information about trials
    file_name = sprintf('data/trialinfo%d.mat',j);
    load(file_name);
%     Concatenate trial information
    curr_end = size(features,1);
    all_trialinfo(indices(j)+1:indices(j)+curr_end) = trialsave;
end

% Remove nans
all_data(~any(~isnan(all_data), 2),:)=[];
all_trialinfo(~any(~isnan(all_trialinfo), 1))=[];

% Store correct values for indices
indices(1) = 1;
indices(end) = size(all_data,1);

% Get labels
labels(all_trialinfo<9)=-1; %Same choice
labels(all_trialinfo>8)=+1; %Different choice
labels = labels.';

% Conditions paired with index values
condition_pairs = [13 14 15 16 9 10 11 12];

% Iterate over all subjects
for ncv = 2:4   
%     Get data and trial information for that current subject
    raw_features_test = all_data(indices(ncv):indices(ncv+1),:);
    raw_testLabel = labels(indices(ncv):indices(ncv+1));
    nosub_test_trialinfo = all_trialinfo(indices(ncv):indices(ncv+1));
    
%     Get data and trial infomation for all subjects other than the current subject
    if ncv ~= 1
        raw_features_train = all_data(1:indices(ncv),:);
        raw_trainLabel = labels(1:indices(ncv));
        nosub_train_trialinfo = all_trialinfo(1:indices(ncv));
        rem_samples = indices(end)-indices(ncv+1);
        raw_features_train(end+1:end+rem_samples,:) = all_data(indices(ncv+1)+1:end,:);
        raw_trainLabel(end+1:end+rem_samples) = labels(indices(ncv+1)+1:end);
        nosub_train_trialinfo(end+1:end+rem_samples) = all_trialinfo(indices(ncv+1)+1:end);
    else
        raw_features_train = all_data(indices(ncv+1)+1:end,:);
        raw_trainLabel = labels(indices(ncv+1)+1:end);
        nosub_train_trialinfo = all_trialinfo(indices(ncv+1)+1:end);
    end
    
%     Iterate over 8 pairs of conditions
    for leave_out = 2:4      
%         Get training data i.e. all data leaving out that of current subject and current condition pair
        trainData = sparse(raw_features_train(nosub_train_trialinfo ~= leave_out | nosub_train_trialinfo ~= condition_pairs(leave_out),:));
        trainLabel = raw_trainLabel(nosub_train_trialinfo ~= leave_out | nosub_train_trialinfo ~= condition_pairs(leave_out));
%         Get testing data i.e. data of current subject and current condition pair
        testData = sparse(raw_features_test(nosub_test_trialinfo == leave_out | nosub_test_trialinfo == condition_pairs(leave_out),:));
        testLabel = raw_testLabel(nosub_test_trialinfo == leave_out | nosub_test_trialinfo == condition_pairs(leave_out)); 
      
%         Obtained by trying out different values and combinations
        param = '-c 1 -g 0.07 -h 0';

%         Train model
        model = svmtrain(trainLabel, trainData, param);
%         Test model
        [~, ~, ~] = svmpredict(testLabel, testData, model);

%         Save model
        temp = sprintf('model_files/model_%d.mat',(ncv-1)*8+leave_out);
        save(temp,'model');
%         Obtain weights and biases of the SVM model
        wts = model.SVs' * model.sv_coef;
        bs = -model.rho;
        if model.Label(1)==-1
            wts = -wts;
            bs = -bs;
        end
%         Save weights
        temp = sprintf('model_files/weights_%d.mat',(ncv-1)*8+leave_out);
        save(temp,'wts');
%         Save biases
        temp = sprintf('model_files/bias_%d.mat',(ncv-1)*8+leave_out);
        save(temp,'bs');
    end
end