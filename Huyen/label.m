function result = label (data,c,classT)
    [d,p] = size(data);  % d = number of records
    eegclass = zeros(d,c);
    for i = 1:d
        for j = 1:c
            if (data{i,26}==classT(j))
                eegclass(i,j) = 1;
            end
        end
    end
    
    result = eegclass;
%     input = transpose(data)
%     target = transpose(eegclass)

end