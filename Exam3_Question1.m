%% ECE 5644 Exam 3 Question 1
% by Richard Cole
clear all; close all;

%% generate datasets
NumClasses = 3;
NumDim = 2;
NumTrainSets = 3;
[dataTrain100,labelsTrain100] = generateMultiringDataset(NumClasses,100);
[dataTrain500,labelsTrain500] = generateMultiringDataset(NumClasses,500);
[dataTrain1000,labelsTrain1000] = generateMultiringDataset(NumClasses,1000);
[dataTest,labelsTest] = generateMultiringDataset(NumClasses,10000);

% create arrays to hold label probabilities for training
pmfTrain100 = full(ind2vec(labelsTrain100,NumClasses));
pmfTrain500 = full(ind2vec(labelsTrain500,NumClasses));
pmfTrain1000 = full(ind2vec(labelsTrain1000,NumClasses));
pmfTest = full(ind2vec(labelsTest,NumClasses));

%% Train Neural Nets with K-fold Cross-Validation

% Split Datasets for k-fold validation
K = 10;
% Note: this method would not work if the number of samples was not
% divisible by 10
dataTrain100Fold = reshape(dataTrain100,NumDim,[],K);
pmfTrain100Fold = reshape(pmfTrain100,NumClasses,[],K);
labelsTrain100Fold = reshape(labelsTrain100,1,[],K);
dataTrain500Fold = reshape(dataTrain500,NumDim,[],K);
pmfTrain500Fold = reshape(pmfTrain500,NumClasses,[],K);
labelsTrain500Fold = reshape(labelsTrain500,1,[],K);
dataTrain1000Fold = reshape(dataTrain1000,NumDim,[],K);
pmfTrain1000Fold = reshape(pmfTrain1000,NumClasses,[],K);
labelsTrain1000Fold = reshape(labelsTrain1000,1,[],K);

% Set up performance arrays
MaxNeurons = 8;
NumFolds = 10;
MaxEpoch = 70000;

%perf1=zeros(NumFolds,MaxNeurons);
perf2=zeros(NumFolds,MaxNeurons);
perf3=zeros(NumFolds,MaxNeurons);

w1_3 = cell(NumFolds,MaxNeurons);
w2_3 = cell(NumFolds,MaxNeurons);
b2_3 = cell(NumFolds,MaxNeurons);
b1_3 = cell(NumFolds,MaxNeurons);

w1_2 = cell(NumFolds,MaxNeurons);
w2_2 = cell(NumFolds,MaxNeurons);
b2_2 = cell(NumFolds,MaxNeurons);
b1_2 = cell(NumFolds,MaxNeurons);

% iterate over different layer sizes
for L=3:MaxNeurons
    
    % perform k-fold cross validation
    for k=1:NumFolds
        % set up training and validation data for this fold
        d_val = dataTrain1000Fold(:,:,k);
        d_val_prob = pmfTrain1000Fold(:,:,k);
        d_val_labels = labelsTrain1000Fold(:,:,k);
        
        % remove the k row and reshape into single column vector
        d_train = dataTrain1000Fold;
        d_train_prob = pmfTrain1000Fold;
        d_train(:,:,k) = [];
        d_train_prob(:,:,k) = [];
        d_train = reshape(d_train,NumDim,[]);
        d_train_prob = reshape(d_train_prob,NumClasses,[]);
        
        % Set up clean neural net
        % number of perceptrons in each layer
        numPerLay1=L;
        
        % create basic  MATLAB patternnet as base
        %already has sinusoidal first layer and softmax out
        net3 = patternnet(numPerLay1,'traingd');
        % set net 3 initilization to random
        net3.initFcn = 'initlay';
        net3.layers{1}.initFcn = 'initwb';
        net3.layers{2}.initFcn = 'initwb';
        net3.inputWeights{1,1}.initFcn = 'rands';
        net3.layerWeights{1,1}.initFcn = 'rands';
        net3.biases{1}.initFcn = 'rands';
        net3.biases{2}.initFcn = 'rands';
        net3.trainParam.epochs = MaxEpoch;
        
        % create basic  MATLAB patternnet as base
        net2 = patternnet(numPerLay1,'traingd');
        %change first layer to softrelu, already has softmax out
        net2.layers{1}.transferFcn='softrelu';
        
        % set net 3 initilization to random
        net2.initFcn = 'initlay';
        net2.layers{1}.initFcn = 'initwb';
        net2.layers{2}.initFcn = 'initwb';
        net2.inputWeights{1,1}.initFcn = 'rands';
        net2.layerWeights{1,1}.initFcn = 'rands';
        net2.biases{1}.initFcn = 'rands';
        net2.biases{2}.initFcn = 'rands';
        net2.trainParam.epochs = MaxEpoch;
        
        % configure nets
        net2 = configure(net2,d_train,d_train_prob);
        net3 = configure(net3,d_train,d_train_prob);
        
        % initilize nets
        net2 = init(net2);
        net3 = init(net3);
        
        % check networks create properly
        %view(net2);
        %view(net3);
        
        % train networks
        [net2,tr2(k,L)] = train(net2,d_train,d_train_prob,'showResources','yes');
        [net3,tr3(k,L)] = train(net3,d_train,d_train_prob,'showResources','yes');
        
        % put validation data through network
        y2 = net2(d_val);
        y3 = net3(d_val);
        
        % convert from pmf to classes
        [M2,I2] = max(y2,[],1);
        [M3,I3] = max(y3,[],1);
        
        % calculate percent correct
        perf2(k,L) = sum(I2==d_val_labels)/numel(d_val_labels);
        perf3(k,L) = sum(I3==d_val_labels)/numel(d_val_labels);
        
        % record weight and biases of trained net
        w1_3(k,L) = net3.IW(1,1);
        w2_3(k,L) = net3.LW(2,1);
        b1_3(k,L) = net3.b(1);
        b2_3(k,L) = net3.b(2);
        
        w1_2(k,L) = net2.IW(1,1);
        w2_2(k,L) = net2.LW(2,1);
        b1_2(k,L) = net2.b(1);
        b2_2(k,L) = net2.b(2);
    end
    
end
%% Pull out best parameters from training

[MPerf2,IPerf2] = max(perf2(:));
[MPerf3,IPerf3] = max(perf2(:));

[optInit2,optNeurons2] = ind2sub(size(perf2),IPerf2);
[optInit3,optNeurons3] = ind2sub(size(perf3),IPerf3);

%% Retrain networks with optimal parameters

MaxFinalEpoch = 50000;

% create basic  MATLAB patternnet as base
net3 = patternnet(optNeurons3,'traingd');
% set net 3 initilization to random
net3.initFcn = 'initlay';
net3.layers{1}.initFcn = 'initwb';
net3.layers{2}.initFcn = 'initwb';
net3.inputWeights{1,1}.initFcn = 'rands';
net3.layerWeights{1,1}.initFcn = 'rands';
net3.biases{1}.initFcn = 'rands';
net3.biases{2}.initFcn = 'rands';
net3.trainParam.epochs = MaxFinalEpoch;

% create basic  MATLAB patternnet as base
net2 = patternnet(optNeurons2,'traingd');
%change first layer to softrelu, already has softmax out
net2.layers{1}.transferFcn='softrelu';

% set net 3 initilization to random
net2.initFcn = 'initlay';
net2.layers{1}.initFcn = 'initwb';
net2.layers{2}.initFcn = 'initwb';
net2.inputWeights{1,1}.initFcn = 'rands';
net2.layerWeights{1,1}.initFcn = 'rands';
net2.biases{1}.initFcn = 'rands';
net2.biases{2}.initFcn = 'rands';
net2.trainParam.epochs = MaxFinalEpoch;

% configure nets
net2 = configure(net2,dataTrain100,pmfTrain100);
net3 = configure(net3,dataTrain100,pmfTrain100);

% initilize nets
net2 = init(net2);
net3 = init(net3);

%change nets initial values to optimal values from training networks
net2.IW(1,1) = w1_2(optInit2,optNeurons2);
net2.b(1) = b1_2(optInit2,optNeurons2);
net2.LW(2,1) = w2_2(optInit2,optNeurons2);
net2.b(2) = b2_2(optInit2,optNeurons2);

net3.IW(1,1) = w1_3(optInit3,optNeurons3);
net3.b(1) = b1_3(optInit3,optNeurons3);
net3.LW(2,1) = w2_3(optInit3,optNeurons3);
net3.b(2) = b2_3(optInit3,optNeurons3);

% check networks create properly
view(net2);
view(net3);

% train networks
%net1 = train(net1,d_train,d_train_prob);
[net2,trf2] = train(net2,dataTrain100,pmfTrain100,'showResources','yes');
[net3,trf3] = train(net3,dataTrain100,pmfTrain100,'showResources','yes');

%% Test Network with Test Data
% put Test Data through network
y2 = net2(dataTest);
y3 = net3(dataTest);

% convert from pmf to classes
[M2,I2] = max(y2,[],1);
[M3,I3] = max(y3,[],1);

% calc final performances
finalPerf2 = sum(I2==labelsTest)/numel(labelsTest)
finalPerf3 = sum(I3==labelsTest)/numel(labelsTest)

