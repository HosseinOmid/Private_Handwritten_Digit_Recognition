%% Skript zum Klassifizieren von handgeschribenen Zahlen zw. 0-9
% Verfasser:        Hossein Omid Beiki 
% Versionshistorie:
% V1     04.03.2019  initialversion
% ------------------------------------------------------------------------
clc
clear

%% load the data
% MNIST is a free data set available in internet
trainImages = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
% load the test data set
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% go through the data set, each column represents an image of size 28x28 pixels
% convert 1D matix to 2D
trainPix = zeros(28,28,size(trainImages,2));
for i=1: size(trainImages,2)
    for idx = 1 : size(trainImages,1)
        y = ceil(idx/28);
        x = idx - (y-1)*28;
        trainPix(x,y,i) = trainImages(idx,i);
    end
end
% go through the test dataset and preprocess them
testPix = zeros(28,28,size(testImages,2));
for i=1: size(testImages,2)
    for idx = 1 : size(testImages,1)
        y = ceil(idx/28);
        x = idx - (y-1)*28;
        testPix(x,y,i) = testImages(idx,i);
    end
end

%% campute the hu-moments
features_train = calc_hu_moment(trainPix);
features_test = calc_hu_moment(testPix);

%% train & test a KNN classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    features_train{:,:}, ...
    trainLabels, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1; 2; 3; 4; 5; 6; 7; 8; 9]);
% Perform cross-validation
partitionedModel = crossval(classificationKNN, 'KFold', 5);
% Compute validation predictions & validation accuracy
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
knn_validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
% compute train predictions & train accuracy
knn_trainPrediction = predict(classificationKNN,features_train{:,:});
knn_trainAccuracy = length(find(knn_trainPrediction == trainLabels)) / length(trainLabels);
% compute test predictions & test accuracy
knn_testPrediction = predict(classificationKNN,features_test{:,:});
knn_testAccuracy = length(find(knn_testPrediction == testLabels)) / length(testLabels);
% show the values of train, validation & test accuracy
disp(['Train-Accuracy of KNN is: ' num2str(knn_trainAccuracy)]);
disp(['Validation-Accuracy of KNN is: ' num2str(knn_validationAccuracy)]);
disp(['Test-Accuracy of KNN is: ' num2str(knn_testAccuracy)]);
disp('compare the train, validation and test accuracy to diagnose the model');

%%  Train a SVM classifier 
% This code specifies all the classifier options and trains the classifier.
rndperm = randperm (RandStream('mt19937ar','Seed',0),size(trainLabels,1));
template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 4, ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    features_train{:,:}, ...
    trainLabels, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', [0; 1; 2; 3; 4; 5; 6; 7; 8; 9]);
% compute train predictions & train accuracy
svm_trainPrediction = predict(classificationSVM, features_train{:,:});
svm_trainAccuracy = length(find(svm_trainPrediction == trainLabels)) / length(trainLabels);
% Compute test predictions & test accuracy
svm_testPrediction = predict(classificationSVM, features_test{:,:});
svm_testAccuracy = length(find(svm_testPrediction == testLabels)) /  length(testLabels);
% show the values of train & test accuracy
disp(['Train-Accuracy of SVM is: ' num2str(svm_trainAccuracy)]);
disp(['Test-Accuracy of SVM is: ' num2str(svm_testAccuracy)]);
disp('compare the train and test accuracy to diagnose the model');

%% show the image and the prediction
for i=1:20
    prd = predict(classificationKNN, features_test{i,:});
    lbl = testLabels(i);
    im = (testPix(:,:,i));
    h1 = imshow(im);
    if lbl~=prd
        cl = 'r';
        prt ='FALSE';
    else
        cl = 'b';
        prt='TRUE';
    end
    title([prt ' Predict: ' num2str(prd)  'Label: ' num2str(lbl)] ,'Color', cl)
    keyboard
end

%% fuction calculate the hu moments
function feat = calc_hu_moment(pixArr)
% https://en.wikipedia.org/wiki/Image_moment
for i = 1 : size(pixArr,3)
    pixArrXY = pixArr(:,:,i);
    
    x = repmat([1:28],28,1);
    y = repmat([1:28]',1,28);
    
    M00 = sum(pixArrXY,'all');
    M11 = sum(x.*y.*pixArrXY,'all');
    M10 = sum(x.*pixArrXY,'all');
    M01 = sum(y.*pixArrXY,'all');
    M20 = sum(x.*x.*pixArrXY,'all');
    M02 = sum(y.*y.*pixArrXY,'all');
    M22 = sum(x.*x.*y.*y.*pixArrXY,'all');
    M21 = sum(x.*x.*y.*pixArrXY,'all');
    M12 = sum(x.*y.*y.*pixArrXY,'all');
    M30 = sum(x.*x.*x.*pixArrXY,'all');
    M03 = sum(y.*y.*y.*pixArrXY,'all');
    
    x_c = M10/M00;
    y_c = M01/M00;
    
    mu00 = M00;
    %mu01 = 0;
    %mu10 = 0;
    mu11 = M11 - x_c*M01;
    mu20 = M20 - x_c*M10;
    mu02 = M02 - y_c*M01;
    mu21 = M21 - 2*x_c*M11;
    mu12 = M12 - 2*y_c*M11;
    mu03 = M03 - 3*y_c*M02 + 2*y_c^2*M01;
    mu30 = M30 - 3*x_c*M20 + 2*x_c^2*M10;
    
    n11 = mu11/mu00^2;
    n20 = mu20/mu00^2;
    n02 = mu02/mu00^2;
    n21 = mu21/mu00^2;
    n12 = mu12/mu00^2;
    n03 = mu03/mu00^2;
    n30 = mu30/mu00^2;
    
    
    I1 = n20 + n02;
    I2 = (n20 - n02)^2 + 4*n11^2;
    I1mu00 = I1*mu00^2;
    I2mu00 = I2*mu00^2;
    I3 = (n30 - 3*n12)^2 + (3*n21 - n03)^2;
    I4 = (n30 + n12)^2 + (n21 + n03)^2;
    I7 = (3*n21 - n03) * (n30 + n12) * ((n30 + n12)^2 - 3*(n21 + n03)^2) - ...
        (n30 - 3*n12) * (n21 + n03) * (3*(n30 + n12)^2 - (n21 + n03)^2);
    
    if i==1
        tb = table(M00, M11, M10, M01, M20, M02, M21, M12, M22, M03, M30, ...
            mu11, mu20, mu02, mu21, mu12, mu03, mu30, ...
            n11, n20, n02, n21, n12, n30, n03, ...
            I1, I2, I3, I4, I7);
        varNames = tb.Properties.VariableNames;
        feat = zeros(size(pixArr,3),size(tb,2));
        feat(1,:) = tb{1,:};
    else
        feat(i,:) = [M00, M11, M10, M01, M20, M02, M21, M12, M22, M03, M30, ...
            mu11, mu20, mu02, mu21, mu12, mu03, mu30, ...
            n11, n20, n02, n21, n12, n30, n03, ...
            I1, I2, I3, I4, I7];
    end    
end
feat = array2table(feat,'VariableNames', varNames);
end