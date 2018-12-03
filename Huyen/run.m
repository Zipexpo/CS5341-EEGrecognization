classT = unique(class.mainclass)
size(classT)
[c,p] = size(classT)
[d,p] = size(data)

eegclass = zeros(d,c);
for i = 1:d
    for j = 1:c
        if (class{i,:}==classT(j))
            eegclass(i,j) = 1;
        end
    end
end
input = transpose(data)
target = transpose(eegclass)

netP = patternnet(10);

net = patternnet([10],'traingd')
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';

%  ================

net = patternnet([10,10]);
net.layers{1}.transferFcn = 'poslin';
net.divideFcn = 'divideind';
net.divideParam.trainInd = (1:1406);
net.divideParam.valInd = (1407:1757);
net.divideParam.testInd = (1758:2196);
[net,tr] = train(net,newInput,newTarget);









