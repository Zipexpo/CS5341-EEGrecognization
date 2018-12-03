function std = standardize(data,m,s)
    [r,c] = size(data);
    for i = 1:c
        for j = 1:r
            std(j,i) = (data(j,i) - m(1,i))/s(1,i);
        end
    end
end