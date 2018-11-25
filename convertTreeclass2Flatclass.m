function [a] = convertTreeclass2Flatclass(data)
a = data;
[n,va] = size(a);
for i = 1:n
    if isundefined(a.class4(i))
        if isundefined(a.class3(i))
            if isundefined(a.class2(i))
                a.mainclass(i) = a.class1(i);
            else
                a.mainclass(i) = a.class2(i);
            end
        
        else
            a.mainclass(i) = a.class3(i);
        end
    else
        a.mainclass(i) = a.class4(i);
    end
end
a.class1 =[];
a.class2 =[];
a.class3 =[];
a.class4 =[];
end