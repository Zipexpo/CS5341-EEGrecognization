
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/kmursi/Desktop/My Pattern Recognetion/Final Project/Final_Filteration_normalized.csv')
f = open('NN_Output.csv', 'w+')
print(data.shape)

feature_names = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'Era_1', 'Era_2', 'Era_3', 'Era_4', 'Era_5',
                 'Erd_1', 'Erd_2', 'Erd_3', 'Erd_4', 'Erd_5', 'Erd_6', 'Erd_7', 'Erd_8',
                 'Erd_9', 'Erd_10', 'Erd_11', 'Erd_12', 'Erd_13', 'Erd_14', 'Erd_15']
solverV=['lbfgs', 'sgd', 'adam']
activationV=['identity', 'logistic', 'tanh', 'relu']
hidden1=[15,25,30,40]
hidden2=[1,15,25,30,40]
PCAV=['on','off']
class_name=['class1','class2','class3','class4']
i_s=0; i_a=0; i_h1=0; i_h2=0; i_p=0; i_c=0
for s in solverV:
    i_a = 0;
    i_h1 = 0;
    i_h2 = 0;
    i_p = 0;
    i_c = 0
    for a in activationV:
        i_h1 = 0;
        i_h2 = 0;
        i_p = 0;
        i_c = 0
        for h1 in hidden1:
            i_h2 = 0;
            i_p = 0;
            i_c = 0
            for h2 in hidden2:
                i_p = 0;
                i_c = 0
                for p in PCAV:
                    i_c = 0
                    for c in class_name:

                        class_names = class_name[i_c]

                        # dataset split to train 75% and test 25%
                        from sklearn.model_selection import train_test_split

                        X = data[feature_names]
                        y = data[class_names]

                        from sklearn.preprocessing import StandardScaler

                        scaler = StandardScaler()
                        # Fit only to the training data
                        scaler.fit(X)
                        X = scaler.transform(X)

                        from sklearn.decomposition import PCA

                        if PCAV[i_p == 'on']:
                            pca = PCA(n_components=22)
                            X = pca.fit_transform(X)

                        X_train, X_test, tr_r, ts_r = train_test_split(X, y, train_size=.8)

                        from sklearn.neural_network import MLPClassifier

                        clf = MLPClassifier(hidden_layer_sizes=(hidden1[i_h1], hidden2[i_h2]), max_iter=5000,
                                            activation=activationV[i_a],
                                            solver=solverV[i_s])
                        clf.fit(X_train,tr_r)
                        solverV = ['lbfgs', 'sgd', 'adam']
                        activationV = ['identity', 'logistic', 'tanh', 'relu']
                        hidden1 = [15, 25, 30, 40]
                        hidden2 = [1, 15, 25, 30, 40]
                        PCAV = ['on', 'off']
                        class_name = ['class1', 'class2', 'class3', 'class4']
                        line_to_write = str(solverV[i_s]) + ', ' + str(activationV[i_a]) + ',' + str(hidden1[i_h1]) + ', ' + str(hidden2[i_h2]) + ', ' + str(
                            PCAV[i_p]) + ', ' + str(class_name[i_c]) + ', ' + str(clf.score(X_train, tr_r)) + ', ' + str(
                            clf.score(X_test, ts_r))+'\n'

                        f.write(line_to_write)  # Give your csv text here.
                        ## Python will convert \n to os.linesep


                        i_c +=1
                    i_p +=1
                i_h2 +=1
            i_h1 +=1
        i_a +=1
    i_s +=1

f.close()