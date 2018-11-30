

results = trainedModel.predictFcn(testingdata);
racc = sum(results == testingdata.mainclass);
sizetest = size(results);
sizetest = sizetest(1);
accuracy = racc/sizetest;