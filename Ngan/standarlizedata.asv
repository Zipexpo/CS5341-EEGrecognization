arraytrain = table2array(traindata(:,1:25));
[Ztrain,mu,sigma] = zscore(arraytrain);
traindata(:,1:25) = array2table(Ztrain,'VariableNames',traindata.Properties.VariableNames(1:25))


%% testing standalize

arraytest= (testdata{:,1:25}-mu)./sigma
testdata(:,1:25) = array2table(arraytest,'VariableNames',testdata.Properties.VariableNames(1:25));
%%
results = trainedModelN.predictFcn(testdata);
racc = sum(results == testdata.mainclass);
        sizetest = size(results);
        sizetest = sizetest(1);
        testingA = racc/sizetest;