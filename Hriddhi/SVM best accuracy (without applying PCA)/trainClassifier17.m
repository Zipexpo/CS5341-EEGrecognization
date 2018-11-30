function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% Extract predictors and response
inputTable = trainingData;
predictorNames = {'E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'Era_1', 'Era_2', 'Era_3', 'Era_4', 'Era_5', 'Erd_1', 'Erd_2', 'Erd_3', 'Erd_4', 'Erd_5', 'Erd_6', 'Erd_7', 'Erd_8', 'Erd_9', 'Erd_10', 'Erd_11', 'Erd_12', 'Erd_13', 'Erd_14', 'Erd_15'};
predictors = inputTable(:, predictorNames);
response = inputTable.class4;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
template = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 100, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', [1; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'Era_1', 'Era_2', 'Era_3', 'Era_4', 'Era_5', 'Erd_1', 'Erd_2', 'Erd_3', 'Erd_4', 'Erd_5', 'Erd_6', 'Erd_7', 'Erd_8', 'Erd_9', 'Erd_10', 'Erd_11', 'Erd_12', 'Erd_13', 'Erd_14', 'Erd_15'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2018b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
inputTable = trainingData;
predictorNames = {'E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'Era_1', 'Era_2', 'Era_3', 'Era_4', 'Era_5', 'Erd_1', 'Erd_2', 'Erd_3', 'Erd_4', 'Erd_5', 'Erd_6', 'Erd_7', 'Erd_8', 'Erd_9', 'Erd_10', 'Erd_11', 'Erd_12', 'Erd_13', 'Erd_14', 'Erd_15'};
predictors = inputTable(:, predictorNames);
response = inputTable.class4;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
