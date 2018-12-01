%loopsetting = 10:100:2000;
loopsetting = [10:100:2000 2200:200:5000];
LearningCyclessetting = 10:10:50;
count = 1;
count2 = 1;
testingAccuracy = zeros(length(loopsetting),length(LearningCyclessetting));
timeper = testingAccuracy;
validationAccuracy = testingAccuracy;
for i = loopsetting
    for j = LearningCyclessetting
        tic
        [trainedClassifier, validationAccuracy(count,count2)] = baggedtree(traindata,[i 30]);
        timeper(count,count2) = toc;
        results = trainedClassifier.predictFcn(testdata);
        racc = sum(results == testdata.mainclass);
        sizetest = size(results);
        sizetest = sizetest(1);
        testingAccuracy(count,count2) = racc/sizetest;
        count2 = count2+1;
    end
    count2 =1;
    count = count+1;
end
save('result.mat', 'timeper', 'testingAccuracy','validationAccuracy','loopsetting','LearningCyclessetting');


%% 
figure(1)
plot(loopsetting,validationAccuracy)