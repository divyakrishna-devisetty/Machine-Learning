"""
manual k fold and shuffle of skutils.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from collections import Counter, defaultdict
from sklearn.utils import shuffle
import itertools


DATA_DIR = "./data/glass.csv"

COEF_ = 'coef0'
DEGREE = 'degree'
R = 'r'
GAMMA = 'gamma'
C_VALUE = 'c_value'



data = pd.read_csv("./data/glass.csv")

g = [2 ** i for i in range(-15, 3)]
c = [2 ** i for i in range(-5, 16)]
r=[2**i for i in range(-15,3)]
r_sigma=[2**i for i in range(-5,3)]
grid_parameters = {
    'linear': {
        # C_VALUE: [1, 10, 100, 1000, 10000]
        C_VALUE:c
    },
    'rbf': {
        # GAMMA: [10 ** -8, 10 ** -4, 1e-3, 2e-3, 5e-3, 5e-2, 8e-2, 9e-2,2**-15,2**-2, 2**-3,2**-1,2**-5,2**-10, 0.1, 0.3, 0.5, 1, 2, 4, 8, 10],
        GAMMA: g,

        # C_VALUE: [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,2**-5, 100,2**8,2**10,2**20, 1000, 2000, 3000, 4000, 10 ** 4,
        #           10 ** 5, 10 ** 6, 10 ** 7]
        C_VALUE:c
    },
    'poly': {
        # C_VALUE: [2**-8,1, 10, 100, 1000, 10 ** 4],
        # GAMMA: [1e-3, 1e-4, 0.05, 0.09,2**-1],
        GAMMA:g,
        C_VALUE:c,
        DEGREE: [2, 3, 4, 5],
        # COEF_: [4,10 ** -5]
        COEF_:r
        # COEF_: [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10, 10 ** 1, 100, 1000, 10000]
    },
    'sigmoid': {
        # C_VALUE: [1,  10, 100, 1000, 10 ** 4],
        C_VALUE:c,
        # GAMMA: [1e-3, 0.00001, 0.002, 0.005, 0.004, 1e-4, 0.05,  0.1, 0.5, 1],
        GAMMA:g,
        COEF_: r_sigma
    }
}
# m = data.sample(frac=1, random_state=1).reset_index(drop=True)
train_features = data.iloc[:, :-1]
test = data.iloc[:, -1]

train_features = train_features.values
test = test.values
# print("before shuffle")
# print(train_features)
# print(test)
# print(type(train_features), type(test))

train_features, y = shuffle(train_features, test, random_state=1)
# print("after shuffle")
# print(train_features)
# print(y)

sc = StandardScaler()
X = sc.fit_transform(train_features)


def get_class_weights(y):
    label_count_map = Counter(y)
    # print(label_count_map)
    total = sum(label_count_map.values())
    weights_dict = {}
    for i in label_count_map:
        weights_dict[i] =  total/ label_count_map[i]
    return weights_dict


def get_folds():
    X_folds = []
    y_folds = []
    for i in range(0, len(X), 43):
        X_folds.append(X[i:i + 43])
        y_folds.append(y[i:i + 43])
    # print(len(X_folds), len(X_folds[0]), len(X_folds[4]))
    # print(len(y_folds), len(y_folds[0]), len(y_folds[4]))
    return X_folds, y_folds


fold_X, fold_y = get_folds()


def linear_svm(kernel, df='ovo', cw=False):
    print("Running svm for linear kernel", kernel)
    print()
    # k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # k_fold = KFold(5)
    fold_accuracies = []
    # for k, (train, test) in enumerate(k_fold.split(X, y)):
    for k in range(5):
        X_data = fold_X[:]
        y_data = fold_y[:]
        del X_data[k]
        del y_data[k]
        X_train_data = list(itertools.chain.from_iterable(X_data))
        X_test_data = fold_X[k]
        y_train_data = list(itertools.chain.from_iterable(y_data))
        y_test_data = fold_y[k]

        print("Running fold:", k + 1)
        # train_data = X[train], y[train]
        # test_data = X[test], y[test]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_data, y_train_data, test_size=0.20, random_state=1)
        # print('{0:15}    {1:2} {2:5}'.format('kernel', 'C', 'validation_accuracy'))
        max_val_accuracy = 0
        best_params = None
        for c_value in grid_parameters[kernel][C_VALUE]:
            if df == 'ovr':
                clf = OneVsRestClassifier(SVC(kernel=kernel, C=c_value, decision_function_shape=df))

            else:
                if cw:
                    clf = SVC(kernel=kernel, C=c_value, decision_function_shape=df,
                              class_weight=get_class_weights(y_train))
                    # clf = SVC(kernel=kernel, C=c_value, decision_function_shape=df,
                    #           class_weight='balanced')
                else:
                    clf = SVC(kernel=kernel, C=c_value, decision_function_shape=df)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            a = np.sum(np.asarray(y_pred) == np.asarray(y_val))
            val_accuracy = (a / len(y_val)) * 100
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                best_params = c_value
            # print(confusion_matrix(y_val, y_pred))
            # print(classification_report(y_val, y_pred))
            # print('{0:10} ==> {1:5}  {2:5}'.format(kernel, c_value, val_accuracy))
        print()
        print("Best validation accuracy:", max_val_accuracy)
        print()
        if df == 'ovr':
            new_model = OneVsRestClassifier(SVC(kernel='linear', C=best_params, decision_function_shape=df))
        else:
            if cw:
                new_model = SVC(kernel='linear', C=best_params, decision_function_shape=df,
                                class_weight=get_class_weights(y_train_data))
                # new_model = SVC(kernel='linear', C=best_params, decision_function_shape=df,
                #                 class_weight='balanced')
            else:
                new_model = SVC(kernel='linear', C=best_params, decision_function_shape=df)
        new_model.fit(X_train_data, y_train_data)
        total_preds = new_model.predict(X_test_data)
        total_crct_preds = np.sum(np.asarray(total_preds) == np.asarray(y_test_data))
        fold_accuracy = (total_crct_preds / len(total_preds)) * 100
        fold_accuracies.append(fold_accuracy)
        print()
        print(
            'Fold :{0:3} ==> Test Accuracy : {1:3}  Optimal Params: C: {2:5}'.format(k + 1, fold_accuracy, best_params))
        print()
    avg_acc = sum(fold_accuracies) / 5
    print()
    print("Classification accuracy for linear kernel:", avg_acc)
    # print("optimal hyperparameters for linear kernel are C: 10")
    print()


def rbf_svm(kernel, df='ovo', cw=False):
    print("Running svm for rbf kernel", kernel)
    print()
    # k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # k_fold = KFold(5)
    fold_accuracies = []
    # for k, (train, test) in enumerate(k_fold.split(X, y)):
    for k in range(5):
        X_data = fold_X[:]
        y_data = fold_y[:]
        del X_data[k]
        del y_data[k]
        X_train_data = list(itertools.chain.from_iterable(X_data))
        X_test_data = fold_X[k]
        y_train_data = list(itertools.chain.from_iterable(y_data))
        y_test_data = fold_y[k]

        print("Running fold:", k + 1)
        # train_data = X[train], y[train]
        # test_data = X[test], y[test]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_data, y_train_data, test_size=0.20, random_state=1)
        # print('{0:15}    {1:2} {2:7} {3:7}'.format('kernel', 'C', 'gamma', 'validation_accuracy'))
        max_val_accuracy = 0
        best_params = None
        max_accuracy_list = defaultdict(list)
        for c_value in grid_parameters[kernel][C_VALUE]:
            for gamma in grid_parameters[kernel][GAMMA]:
                if df == 'ovr':
                    clf = OneVsRestClassifier(SVC(kernel=kernel, C=c_value, gamma=gamma, decision_function_shape=df))
                else:
                    if cw:
                        clf = SVC(kernel=kernel, C=c_value, gamma=gamma, decision_function_shape=df,
                                  class_weight=get_class_weights(y_train))
                        # clf = SVC(kernel=kernel, C=c_value, gamma=gamma, decision_function_shape=df,
                        #           class_weight='balanced')
                    else:
                        clf = SVC(kernel=kernel, C=c_value, gamma=gamma, decision_function_shape=df)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                a = np.sum(np.asarray(y_pred) == np.asarray(y_val))
                val_accuracy = (a / len(y_val)) * 100

                params = [c_value, gamma]
                max_accuracy_list[val_accuracy].append(params)
                if val_accuracy > max_val_accuracy:
                    max_val_accuracy = val_accuracy
                    best_params = params
                # print('{0:10} ==> {1:5}  {2:5}   {3:5}'.format(kernel, c_value, gamma, val_accuracy))
        print()
        print("Best validation accuracy:", max_val_accuracy)
        # print("max_list:", max_accuracy_list[max_val_accuracy])
        print()
        if df == 'ovr':
            new_model = OneVsRestClassifier(
                SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], decision_function_shape=df))
        else:
            if cw:
                # print("test")
                new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], decision_function_shape=df,
                                class_weight=get_class_weights(y_train_data))
                # new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], decision_function_shape=df,
                #                 class_weight='balanced')
            else:
                new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], decision_function_shape=df)
        new_model.fit(X_train_data, y_train_data)
        total_preds = new_model.predict(X_test_data)
        total_crct_preds = np.sum(np.asarray(total_preds) == np.asarray(y_test_data))
        fold_accuracy = (total_crct_preds / len(total_preds)) * 100
        fold_accuracies.append(fold_accuracy)
        print()
        print('Fold :{0:3} ==> Test Accuracy : {1:3}  Optimal Params: C: {2:2} gamma:{3:5}'.format(k + 1, fold_accuracy,
                                                                                                   best_params[0],
                                                                                                   best_params[1]))
        print()
    avg_acc = sum(fold_accuracies) / 5
    print()
    print("Classification accuracy for rbf kernel:", avg_acc)
    # print("optimal hyperparameters for rbf kernel are C: 100 gamma: 0.05")
    print()


def poly_svm(kernel, df='ovo', cw=False):
    print("Running svm for polynomial kernel", kernel)
    print()
    # k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    k_fold = KFold(5)
    fold_accuracies = []
    # for k, (train, test) in enumerate(k_fold.split(X, y)):
    for k in range(5):
        X_data = fold_X[:]
        y_data = fold_y[:]
        del X_data[k]
        del y_data[k]
        X_train_data = list(itertools.chain.from_iterable(X_data))
        X_test_data = fold_X[k]
        y_train_data = list(itertools.chain.from_iterable(y_data))
        y_test_data = fold_y[k]

        print("Running fold:", k + 1)
        # train_data = X[train], y[train]
        # test_data = X[test], y[test]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_data, y_train_data, test_size=0.20, random_state=1)
        # print('{0:15}    {1:2} {2:7} {3:7} {4:7} {5:7}'.format('kernel', 'C', 'gamma', 'degree', 'r',
        #                                                        'validation_accuracy'))
        max_val_accuracy = 0
        best_params = None
        for c_value in grid_parameters[kernel][C_VALUE]:
            for gamma in grid_parameters[kernel][GAMMA]:
                for degree in grid_parameters[kernel][DEGREE]:
                    for r in grid_parameters[kernel][COEF_]:
                        if df == 'ovr':
                            clf = OneVsRestClassifier(
                                SVC(kernel=kernel, C=c_value, gamma=gamma, degree=degree, decision_function_shape=df))
                        else:
                            if cw:
                                clf = SVC(kernel=kernel, C=c_value, gamma=gamma, degree=degree,
                                          decision_function_shape=df, class_weight=get_class_weights(y_train))
                                # clf = SVC(kernel=kernel, C=c_value, gamma=gamma, degree=degree,
                                #           decision_function_shape=df, class_weight='balanced')
                            else:
                                clf = SVC(kernel=kernel, C=c_value, gamma=gamma, degree=degree,
                                          decision_function_shape=df)

                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_val)
                        a = np.sum(np.asarray(y_pred) == np.asarray(y_val))
                        val_accuracy = (a / len(y_val)) * 100
                        params = [c_value, gamma, degree, r]

                        if val_accuracy > max_val_accuracy:
                            max_val_accuracy = val_accuracy
                            best_params = params
                        # print(
                        #     '{0:10} ==> {1:5}  {2:5}   {3:5}   {4:5}   {5:5}'.format(kernel, c_value, gamma, degree, r,
                        #                                                               val_accuracy))
        print()
        print("Best validation accuracy:", max_val_accuracy)
        print()
        if df == 'ovr':
            new_model = OneVsRestClassifier(
                SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], degree=best_params[2],
                    coef0=best_params[3], decision_function_shape=df))
        else:
            if cw:
                new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], degree=best_params[2],
                                coef0=best_params[3], decision_function_shape=df,
                                class_weight=get_class_weights(y_train_data))
                # new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], degree=best_params[2],
                #                 coef0=best_params[3], decision_function_shape=df,
                #                 class_weight='balanced')
            else:
                new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], degree=best_params[2],
                                coef0=best_params[3], decision_function_shape=df)
        new_model.fit(X_train_data, y_train_data)
        total_preds = new_model.predict(X_test_data)
        total_crct_preds = np.sum(np.asarray(total_preds) == np.asarray(y_test_data))
        fold_accuracy = (total_crct_preds / len(total_preds)) * 100
        fold_accuracies.append(fold_accuracy)
        print()
        print(
            'Fold :{0:3} ==> Test Accuracy : {1:3}  Optimal Params: C: {2:2} gamma:{3:5} degree:{4:2}  r: {5:5}'.format(
                k + 1, fold_accuracy, best_params[0], best_params[1], best_params[2], best_params[3]))
        print()
    avg_acc = sum(fold_accuracies) / 5
    print()
    print("Classification accuracy for poly kernel:", avg_acc)
    # print("optimal hyperparameters for poly kernel are C: 1000 gamma: 0.05 degree: 2 r: 1e-05")
    print()


def sigmoid_svm(kernel, df='ovo', cw=False):
    print("Running svm for sigmoid kernel", kernel)
    print()
    # k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    k_fold = KFold(5)
    fold_accuracies = []
    # for k, (train, test) in enumerate(k_fold.split(X, y)):
    for k in range(5):
        X_data = fold_X[:]
        y_data = fold_y[:]
        del X_data[k]
        del y_data[k]
        X_train_data = list(itertools.chain.from_iterable(X_data))
        X_test_data = fold_X[k]
        y_train_data = list(itertools.chain.from_iterable(y_data))
        y_test_data = fold_y[k]

        print("Running fold:", k + 1)
        # train_data = X[train], y[train]
        # test_data = X[test], y[test]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_data, y_train_data, test_size=0.20, random_state=1)
        # print('{0:15}    {1:2} {2:7} {3:7} {4:7}'.format('kernel', 'C', 'gamma', 'r',
        #                                                  'validation_accuracy'))
        max_val_accuracy = 0
        best_params = None
        for c_value in grid_parameters[kernel][C_VALUE]:
            for gamma in grid_parameters[kernel][GAMMA]:
                for r in grid_parameters[kernel][COEF_]:
                    if df == 'ovr':
                        clf = OneVsRestClassifier(
                            SVC(kernel=kernel, C=c_value, gamma=gamma, coef0=r, decision_function_shape=df))
                    else:
                        if cw:
                            clf = SVC(kernel=kernel, C=c_value, gamma=gamma, coef0=r, decision_function_shape=df,
                                      class_weight=get_class_weights(y_train))
                            # clf = SVC(kernel=kernel, C=c_value, gamma=gamma, coef0=r, decision_function_shape=df,
                            #           class_weight='balanced')
                        else:
                            clf = SVC(kernel=kernel, C=c_value, gamma=gamma, coef0=r, decision_function_shape=df)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)
                    a = np.sum(np.asarray(y_pred) == np.asarray(y_val))
                    val_accuracy = (a / len(y_val)) * 100
                    params = [c_value, gamma, r]

                    if val_accuracy > max_val_accuracy:
                        max_val_accuracy = val_accuracy
                        best_params = params
                    # print('{0:10} ==> {1:5}  {2:5}   {3:5}  {4:5}'.format(kernel, c_value, gamma, r, val_accuracy))
        print()
        print("Best validation accuracy:", max_val_accuracy)
        print()
        if df == 'ovr':
            new_model = OneVsRestClassifier(
                SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], coef0=best_params[2],
                    decision_function_shape=df))
        else:
            if cw:
                new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], coef0=best_params[2],
                                decision_function_shape=df, class_weight=get_class_weights(y_train_data))
                # new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], coef0=best_params[2],
                #                 decision_function_shape=df, class_weight='balanced')
            else:
                new_model = SVC(kernel=kernel, C=best_params[0], gamma=best_params[1], coef0=best_params[2],
                                decision_function_shape=df)

        new_model.fit(X_train_data, y_train_data)
        total_preds = new_model.predict(X_test_data)
        total_crct_preds = np.sum(np.asarray(total_preds) == np.asarray(y_test_data))
        fold_accuracy = (total_crct_preds / len(total_preds)) * 100
        fold_accuracies.append(fold_accuracy)
        print()
        print(
            'Fold :{0:3} ==> Test Accuracy : {1:3}  Optimal Params: C: {2:2} gamma:{3:5} r:{4:2}'.format(
                k + 1, fold_accuracy, best_params[0], best_params[1], best_params[2]))
        print()
    avg_acc = sum(fold_accuracies) / 5
    print()
    print("Classification accuracy for sigmoid kernel:", avg_acc)
    # print("optimal hyperparameters for sigmoid kernel are C: 1000 gamma: 0.05 r: 1e-05")
    print()
import time
# sys.stdout=open('out.txt','w')


print("Running One versus One classifiers")
print()

# linear_start = time.time()
# linear_svm('linear', df='ovo')
# print("time taken by linear oneversusone:",time.time()-linear_start)

# rbf_start = time.time()
# rbf_svm('rbf', df='ovo')
# print("time taken by rbf oneversusone:",time.time()-rbf_start)

poly_start = time.time()
poly_svm('poly', df='ovo')
print("time taken by poly oneversusone:",time.time()-poly_start)

# sig_start = time.time()
# sigmoid_svm('sigmoid', df='ovo')
# print("time taken by sig oneversusone:",time.time()-sig_start)

print()
print("Running One versus rest classifiers")
print()

# linear_ovr_start = time.time()
# linear_svm('linear', df='ovr')
# print("time taken by linear oneversusrest:",time.time()-linear_ovr_start)

# rbf_ovr_start = time.time()
# rbf_svm('rbf', df='ovr')
# print("time taken by rbf oneversusrest:",time.time()-rbf_ovr_start)

# poly_ovr_start = time.time()
# poly_svm('poly', df='ovr')
# print("time taken by poly oneversusrest:",time.time()-poly_ovr_start)
#
# sigmoid_ovr_start = time.time()
# sigmoid_svm('sigmoid', df='ovr')
# print("time taken by sigmoid oneversusrest:",time.time()-sigmoid_ovr_start)

print()
print("Running One versus One classifiers with class weights")
print()
# linear_bal=time.time()
# linear_svm('linear', df='ovo', cw=True)
# print("time taken by linear oneversusonebal:",time.time()-linear_bal)

# rbf_bal = time.time()
# rbf_svm('rbf', df='ovo', cw=True)
# print("time taken by rbf oneversusonebal:",time.time()-rbf_bal)

poly_bal=time.time()
poly_svm('poly', df='ovo', cw=True)
print("time taken by poly oneversusonebal:",time.time()-poly_bal)

# sig_bal=time.time()
# sigmoid_svm('sigmoid', df='ovo', cw=True)
# print("time taken by sigmoid oneversusonebal:",time.time()-sig_bal)