% Time series forecasting -ULSTM
clc; clear; close all;
structurePath = "Data\sNormalP3Ade35.mat";
load(structurePath)

for i = 1:size(s,2)
    tableType = s(i).CVAESigWFinal;
    arrayType = table2array(tableType(1:300,:));
    cellType = num2cell(arrayType',2);
    s(i).extendCVAESULSTM = cellType;
end

timeStep = 15;
startPoint = 25;
stepJump = 15;

offset = 0;
for i = 1: timeStep   
    i

    segs=[];  
    for l = 1: size(s,2)
        segs  = [segs ; s(l).extendCVAESULSTM];     
    end
    
    dataTrain = segs(:); 
    lengthDataTrain = length(dataTrain{1,1});

    predictorStSeg = startPoint + offset + stepJump
    predictorEndSeg = lengthDataTrain-stepJump

    targetStSeg = startPoint + 2*stepJump + offset
    targetEndSeg = lengthDataTrain

    for n = 1:numel(dataTrain)
        dataLoop = dataTrain{n};  
        XTrain{n} = dataLoop(:, predictorStSeg : predictorEndSeg);
        TTrain{n} = dataLoop(:, targetStSeg : targetEndSeg);    
    end

    segLengthTarget = size(XTrain{1,1},2)
    segLengthTarget = size(TTrain{1,1},2);
    offset

    layers = [
        sequenceInputLayer(1)
        lstmLayer(128)
        fullyConnectedLayer(1)
        regressionLayer];

    options = trainingOptions("adam", ...
        MaxEpochs=5, ...        
        MiniBatchSize = 128, ...
        Shuffle="never", ...
        Plots="none", ...
        Verbose=0);

    net = trainNetwork(XTrain,TTrain,layers,options);

    for r =1: size(s,2)
        for c= 1:size(s(r).extendCVAESULSTM,1)
            net = resetState(net);
            segLoop = s(r).extendCVAESULSTM{c};            
            X = segLoop(predictorStSeg:predictorEndSeg);
            [net,Z] = predictAndUpdateState(net,X);
            Xt =  Z(end-stepJump+1:end);
            [net,Zt] = predictAndUpdateState(net,Xt);            
            XFinal = [segLoop Zt];
            s(r).extendCVAESULSTM{c} =XFinal;

        end
    end
    offset= offset + 10;
end


structurePath = "Results\sNormalP3Ade35.mat";
save(structurePath,'s')






