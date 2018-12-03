function [traindata,valdata] = divclassval (datain,holdin)
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
            valdata = outerjoin(valdata, tempdata2,'MergeKeys', true);
        catch
            traindata = tempdata1;
            valdata = tempdata2;
        end
    end
    traindata = traindata(randperm(size(traindata, 1)), :);
    valdata = valdata(randperm(size(valdata, 1)), :);
end