function [traindata,testingdata] = divclass (datain,holdin)
    mainclass = unique(datain.mainclass);
    classlength = size(mainclass);
    for i = 1:classlength(1)
        tempdata = datain(datain.mainclass==mainclass(i), :);
        tempdata = tempdata(randperm(size(tempdata, 1)), :); % shuffle
        [n,d] = size(tempdata);
        tempdata1 =  tempdata(1:round(n*holdin),:);
        tempdata2 =  tempdata(round(n*holdin)+1:n,:);
        try
            traindata = outerjoin(traindata, tempdata1,'MergeKeys', true);
            testingdata = outerjoin(testingdata, tempdata2,'MergeKeys', true);
        catch
            traindata = tempdata1;
            testingdata = tempdata2;
        end
    end
    traindata = traindata(randperm(size(traindata, 1)), :);
    testingdata = testingdata(randperm(size(testingdata, 1)), :);
end