__author__ = 'annasinilo'

import pandas as pd
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.calibration import CalibratedClassifierCV


def main():
    train = pd.read_csv('train.csv')
    # print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

    labels = train['target']
    train.drop(['target', 'id'], axis=1, inplace=True)

    # print(train.head())

    ### we need a test set that we didn't train on to find the best weights for combining the classifiers
    sss = StratifiedShuffleSplit(labels, test_size=0.33, random_state=1234)
    for subtrain_indexes, subtest_indexes in sss:
        break

    train_features = train.values
    train_labels = labels.values
    subtrain_features, subtrain_labels = train.values[subtrain_indexes], labels.values[subtrain_indexes]
    subtest_features, subtest_labels = train.values[subtest_indexes], labels.values[subtest_indexes]

    ### building the classifiers
    clfs = []

    ### usually you'd use xgboost and neural nets here
    # logreg = LogisticRegression(solver='liblinear')
    # logreg.fit(subtrain_features, subtrain_labels)
    # print('LogisticRegression LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, logreg.predict_proba(subtrain_features))))
    # print('LogisticRegression LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, logreg.predict_proba(subtest_features))))
    # clfs.append(logreg)

    # logreg2 = LogisticRegression(solver='newton-cg')
    # logreg2.fit(subtrain_features, subtrain_labels)
    # print('LogisticRegression LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, logreg2.predict_proba(subtrain_features))))
    # print('LogisticRegression LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, logreg2.predict_proba(subtest_features))))
    # clfs.append(logreg2)

    # logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    # logreg.fit(subtrain_features, subtrain_labels)
    # print('LogisticRegression LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, logreg.predict_proba(subtrain_features))))
    # print('LogisticRegression LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, logreg.predict_proba(subtest_features))))
    # clfs.append(logreg)

    # knn = KNeighborsClassifier(n_neighbors=10)
    # knn.fit(subtrain_features, subtrain_labels)
    # print('KNeighborsClassifier LogLoss {score} on SubTrain: '.format(score=log_loss(subtrain_labels, knn.predict_proba(subtrain_features))))
    # print('KNeighborsClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, knn.predict_proba(subtest_features))))
    # clfs.append(knn)

    # knn = KNeighborsClassifier(n_neighbors=20)
    # knn.fit(subtrain_features, subtrain_labels)
    # print('KNeighborsClassifier LogLoss {score} on SubTrain: '.format(score=log_loss(subtrain_labels, knn.predict_proba(subtrain_features))))
    # print('KNeighborsClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, knn.predict_proba(subtest_features))))
    # clfs.append(knn)


    # knn = KNeighborsClassifier(n_neighbors=50)
    # knn.fit(subtrain_features, subtrain_labels)
    # print('KNeighborsClassifier LogLoss {score} on SubTrain: '.format(score=log_loss(subtrain_labels, knn.predict_proba(subtrain_features))))
    # print('KNeighborsClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, knn.predict_proba(subtest_features))))
    # clfs.append(knn)

    # rfc = RandomForestClassifier(n_estimators=25, random_state=1234, n_jobs=-1)
    # rfc.fit(subtrain_features, subtrain_labels)
    # print('RFC LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, rfc.predict_proba(subtrain_features))))
    # print('RFC LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, rfc.predict_proba(subtest_features))))
    # clfs.append(rfc)
#
#
    # rfc2 = RandomForestClassifier(n_estimators=50, random_state=1234, n_jobs=-1)
    # rfc2.fit(subtrain_features, subtrain_labels)
    # print('RFC2 LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, rfc2.predict_proba(subtrain_features))))
    # print('RFC2 LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, rfc2.predict_proba(subtest_features))))
    # clfs.append(rfc2)
#
    # rfc3 = RandomForestClassifier(n_estimators=70, random_state=1234, n_jobs=-1)
    # rfc3.fit(subtrain_features, subtrain_labels)
    # print('RFC3 LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, rfc3.predict_proba(subtrain_features))))
    # print('RFC3 LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, rfc3.predict_proba(subtest_features))))
    # clfs.append(rfc3)
#
    # rfc4 = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1)
    # rfc4.fit(subtrain_features, subtrain_labels)
    # print('RFC4 LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, rfc4.predict_proba(subtrain_features))))
    # print('RFC4 LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, rfc4.predict_proba(subtest_features))))
    # clfs.append(rfc4)
#
    # rfc5 = RandomForestClassifier(n_estimators=1000, random_state=1234, n_jobs=-1)
    # rfc5.fit(subtrain_features, subtrain_labels)
    # print('RFC5 LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, rfc5.predict_proba(subtrain_features))))
    # print('RFC5 LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, rfc5.predict_proba(subtest_features))))
    # clfs.append(rfc5)

    # rfc6 = RandomForestClassifier(n_estimators=500, random_state=1234, n_jobs=-1)
    # calibrated_clf = CalibratedClassifierCV(rfc6, method='isotonic', cv=5)
    # calibrated_clf.fit(train_features, train_labels)
    # print('CalibratedClassifierCV LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, calibrated_clf.predict_proba(subtrain_features))))
    # print('CalibratedClassifierCV LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, calibrated_clf.predict_proba(subtest_features))))
    # clfs.append(calibrated_clf)


    #rfc6.fit(subtrain_features, subtrain_labels)
    #print('RFC6 LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, rfc6.predict_proba(subtrain_features))))
    #print('RFC6 LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, rfc6.predict_proba(subtest_features))))
    #clfs.append(rfc6)

#
    # gnb = GaussianNB()
    # gnb.fit(subtrain_features, subtrain_labels)
    # print('GaussianNB LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, gnb.predict_proba(subtrain_features))))
    # print('GaussianNB LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, gnb.predict_proba(subtest_features))))
    # clfs.append(gnb)
#
#     """svc = NuSVC()
#     svc.fit(subtrain_features, subtrain_labels)
#     print('SVC LogLoss {score}'.format(score=log_loss(subtest_labels, svc.predict_proba(subtest_features))))
#     clfs.append(svc)"""
#
    # dtree = DecisionTreeClassifier(max_depth=5)
    # dtree.fit(subtrain_features, subtrain_labels)
    # print('DecisionTreeClassifier LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, dtree.predict_proba(subtrain_features))))
    # print('DecisionTreeClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, dtree.predict_proba(subtest_features))))
    # clfs.append(dtree)

    # dtree = DecisionTreeClassifier(max_depth=10)
    # dtree.fit(subtrain_features, subtrain_labels)
    # print('DecisionTreeClassifier LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, dtree.predict_proba(subtrain_features))))
    # print('DecisionTreeClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, dtree.predict_proba(subtest_features))))
    # clfs.append(dtree)
    #
    # dtree = DecisionTreeClassifier(max_depth=15)
    # dtree.fit(subtrain_features, subtrain_labels)
    # print('DecisionTreeClassifier LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, dtree.predict_proba(subtrain_features))))
    # print('DecisionTreeClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, dtree.predict_proba(subtest_features))))
    # clfs.append(dtree)

    # dtree = DecisionTreeClassifier(max_depth=20)
    # dtree.fit(subtrain_features, subtrain_labels)
    # print('DecisionTreeClassifier LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, dtree.predict_proba(subtrain_features))))
    # print('DecisionTreeClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, dtree.predict_proba(subtest_features))))
    # clfs.append(dtree)


#     gbm = GradientBoostingClassifier(subsample=0.8, n_estimators=10, learning_rate=0.02)
#     gbm.fit(subtrain_features, subtrain_labels)
#     print('GradientBoostingClassifier LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, gbm.predict_proba(subtrain_features))))
#     print('GradientBoostingClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, gbm.predict_proba(subtest_features))))
#     clfs.append(gbm)

#     gbm = GradientBoostingClassifier(subsample=0.8, n_estimators=10, learning_rate=0.02)
#     gbm.fit(subtrain_features, subtrain_labels)
#     print('GradientBoostingClassifier LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, gbm.predict_proba(subtrain_features))))
#     print('GradientBoostingClassifier LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, gbm.predict_proba(subtest_features))))
#     clfs.append(gbm)
#

    # gbm = GradientBoostingClassifier(n_estimators=100)
    # calibrated_gbm = CalibratedClassifierCV(gbm, method='sigmoid', cv=5)
    # calibrated_gbm.fit(train_features, train_labels)
    # print('CalibratedClassifierCV LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, calibrated_gbm.predict_proba(subtrain_features))))
    # print('CalibratedClassifierCV LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, calibrated_gbm.predict_proba(subtest_features))))
    # clfs.append(calibrated_gbm)

    logreg = LogisticRegression(solver='liblinear')
    calibrated_logreg = CalibratedClassifierCV(logreg, method='isotonic', cv=5)
    calibrated_logreg.fit(train_features, train_labels)
    print('CalibratedClassifierCV LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, calibrated_logreg.predict_proba(subtrain_features))))
    print('CalibratedClassifierCV LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, calibrated_logreg.predict_proba(subtest_features))))
    clfs.append(calibrated_logreg)

    logreg.fit(subtrain_features, subtrain_labels)
    print('LogisticRegression LogLoss {score} on SubTrain'.format(score=log_loss(subtrain_labels, logreg.predict_proba(subtrain_features))))
    print('LogisticRegression LogLoss {score} on SubTest'.format(score=log_loss(subtest_labels, logreg.predict_proba(subtest_features))))
    clfs.append(logreg)

    ### finding the optimum weights
    predictions = []
    for clf in clfs:
        predictions.append(clf.predict_proba(subtest_features))

    #the algorithms need a starting value, right not we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [1.0 / len(predictions)]*len(predictions)

    #adding constraints  and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
        return log_loss(subtest_labels, final_prediction)

    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    best_weights = res['x']
    subtrain_predictions = []
    for clf in clfs:
        subtrain_predictions.append(clf.predict_proba(subtrain_features))
    final_prediction = 0
    for weight, prediction in zip(best_weights, subtrain_predictions):
        final_prediction += weight*prediction
    print('Ensemble Score on SubTrain: {score}'.format(score=log_loss(subtrain_labels, final_prediction)))

    print('Best Weights: {weights}'.format(weights=best_weights))

    test = pd.read_csv('test.csv')
    # print("Testing set has {0[0]} rows and {0[1]} columns".format(test.shape))
    # print(test.head())

    test.drop(['id'], axis=1, inplace=True)
    # print(test.head())

    test_predictions = []
    for clf in clfs:
        test_predictions.append(clf.predict_proba(test))
    final_prediction = 0
    for weight, prediction in zip(best_weights, test_predictions):
        final_prediction += weight*prediction
    i = 1
    foutput = open('submission.csv', 'w')
    print >> foutput, 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9'
    for row in final_prediction:
        output = []
        for x in row:
            if x < 0.00000001:
                x = 0.00000001
            output.append(x)
        line = str(i) + "," + ",".join(map(str, output))
        i += 1
        print >> foutput, line

    print "Done"

if __name__ == '__main__':
    main()
