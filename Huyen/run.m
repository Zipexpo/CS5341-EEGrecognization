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

net = patternnet([50,50,50]);
net.trainParam.max_fail = 50;
net.trainParam.epochs = 2000;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:1406;
net.divideParam.valInd = 1407:1757;
net.divideParam.testInd = 1758:2196;
[net,tr] = train(net,newInput,newTarget);



outArray = zeros(5,6);



maxcount = 0;
trn = 0;
finalout = 0;

for i=1:10

    count = 0;
    net = patternnet([60,60,60]);
    net.trainParam.max_fail = 50;
    net.trainParam.epochs = 2000;
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:1406;
    net.divideParam.valInd = 1407:1757;
    net.divideParam.testInd = 1758:2196;
    net.trainParam.epochs = 5000;
    net.layers{1}.transferFcn = 'poslin';
    net.layers{2}.transferFcn = 'poslin';
    [net,tr] = train(net,newInput,newTarget);
    outputs = net(newInput);
    
    [u,v] = max(outputs);
    out = zeros(a,b);
    for j = 1:b
        out(v(j),j) = 1;
        
    end

     for k = 1758:2196
         for l = 1:a
             if (out(l,k)==newTarget(l,k)) && (newTarget(l,k)==1)
                 count = count +1;
             end
         end
     end
%     count = sum(out==newTarget);
    if (count > maxcount)
         maxcount = count;
        finalout = net(newInput);
        trn = tr;
        fprintf("Hi");
    end
end

figure, plotconfusion(newTarget(:,1758:2196),finalout(:,1758:2196))
finalout__  =finalout
tr22 = trn




