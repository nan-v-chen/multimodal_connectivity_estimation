#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import xlwt
import xlrd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_regression, SelectKBest
from sklearn import svm
from scipy import interp
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from MP_NNF import MP_NNF
import logging
import warnings

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
warnings.filterwarnings("ignore")

# read the subject ids
workbook = xlrd.open_workbook("filtered.xlsx", 'r')
sheet_MDD = workbook.sheet_by_index(0)
sheet_HC = workbook.sheet_by_index(1)
subject_MDD = sheet_MDD.col_values(2)[1:]
subject__HC = sheet_HC.col_values(2)[1:]
subjectList = subject_MDD + subject__HC


# generate xy from .mat computed connectivity file
def generate_x_y(feature_name):
    x = np.empty((len(subjectList), 4005))
    for subject_id in range(len(subjectList)):
        subject = subjectList[subject_id]
        data = loadmat("MPNNF/{f}/{sub}.mat".format(f=feature_name, sub=subject))["data"]
        subject_feature = []
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                subject_feature.append(data[i, j])
        x[subject_id] = np.array(subject_feature)
    y = np.asarray([1] * len(subject_MDD) + [0] * len(subject__HC)).astype('float32')
    return x, y


# generate train and test sets from of 10 folds
def generate_train_test(x, y, seed):
    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    train_test_index = skf.split(x, y)
    return train_test_index


def model_evaluate(test_label, res_label):
    tp = tn = fp = fn = 0
    for j in range(len(test_label)):
        if test_label[j] == 0 and res_label[j] == 0:
            tn += 1
        if test_label[j] == 0 and res_label[j] == 1:
            fp += 1
        if test_label[j] == 1 and res_label[j] == 0:
            fn += 1
        if test_label[j] == 1 and res_label[j] == 1:
            tp += 1
    acc = (tp + tn) / (tp + fp + fn + tn)
    sen = tp / (tp + fn)
    spe = tn / (fp + tn)
    return acc, sen, spe


# parameters in MP-NNF
ks = [10, 20, 30, 40, 50, 60, 70, 80]
cs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
if not os.path.exists("Results"):
    os.mkdir("Results")
if not os.path.exists("MPNNF"):
    os.mkdir("MPNNF")
# result matrices
ACC_matrix = np.zeros((len(ks), len(cs)))
AUC_matrix = np.zeros((len(ks), len(cs)))
for k_id in range(len(ks)):
    for c_id in range(len(cs)):
        k = ks[k_id]
        c = cs[c_id]
        featureName = "MPNNF_{k}_{c}".format(k=k, c=c)
        if not os.path.exists("MPNNF/MPNNF_{k}_{c}".format(k=k, c=c)):
            os.mkdir("MPNNF/MPNNF_{k}_{c}".format(k=k, c=c))
        # estimate multi-modal network connectivity using MP-NNF algorithm. Only to modality in this case.
        for subject in subjectList:
            A = np.empty((2, 90, 90))
            A[0] = loadmat("FN/FN_{sub}.mat".format(sub=subject))["data"]
            A[1] = loadmat("SN/SN_{sub}.mat".format(sub=subject))["data"]
            Dc = MP_NNF(A, k=k, c=c, max_iter=200)
            savemat("MPNNF/MPNNF_{k}_{c}/{sub}.mat".format(k=k, c=c, sub=subject), {'data': Dc})
        mean_fpr = np.linspace(0, 1, 100)
        ACCs = []
        SENs = []
        SPEs = []
        AUCs = []
        tprs = []
        contents = []
        # 10-fold cross validation is repeated 10 times
        for iterCount in range(10):
            X, Y = generate_x_y(featureName)
            trainAndTestIndex = generate_train_test(X, Y, seed=iterCount)
            fold = 0
            # 10-fold cross validation
            for trainIndex, testIndex in trainAndTestIndex:  # 十折交叉验证
                X_train, X_test = X[trainIndex], X[testIndex]
                y_train, y_test = Y[trainIndex], Y[testIndex]
                # normalize train and test sets
                mean = np.mean(X_train)
                std = np.std(X_train)
                X_train_nol = (X_train - mean) / std
                X_test_nol = (X_test - mean) / std
                # grid search process for parameters in feature selection and svm
                acc_max = 0
                sen_max = 0
                featureNum_op = 10
                COp = 1
                # the first step of feature selection
                sk1 = SelectKBest(f_regression, k=100)
                sk1.fit(X_train_nol, y_train)
                X_train_nol_sel = sk1.transform(X_train_nol)
                X_test_nol_sel = sk1.transform(X_test_nol)
                selected1 = sk1.get_support()
                # logger.debug("1st feature selection accomplished")
                if not os.path.exists("Results/selected_{f}".format(f=featureName)):
                    os.mkdir("Results/selected_{f}".format(f=featureName))
                savemat("Results/selected_{f}/selected1_{c}_{fold}.mat".format(f=featureName, c=iterCount, fold=fold), {'data': selected1})
                for feature_num in range(10, X_train_nol_sel.shape[1], 1):
                    for C in np.logspace(-4, 4, 9, base=2):
                        # the second step feature selection
                        lr = LinearRegression()
                        rfe = RFE(lr, n_features_to_select=feature_num)
                        rfe.fit(X_train_nol_sel, y_train)
                        selected2 = rfe.support_
                        # svm classification
                        clf = svm.SVC(kernel='linear', C=C).fit(X_train_nol_sel[:, selected2], y_train)
                        score = clf.score(X_test_nol_sel[:, selected2], y_test)
                        y_score = clf.decision_function(X_test_nol_sel[:, selected2])
                        res = clf.predict(X_test_nol_sel[:, selected2])
                        ACC, SEN, SPE = model_evaluate(test_label=y_test, res_label=res)
                        if score > acc_max:
                            COp = C
                            featureNum_op = feature_num
                            acc_max = score
                            sen_max = SEN
                        if score == acc_max:
                            if SEN > sen_max:
                                COp = C
                                featureNum_op = feature_num
                                acc_max = score
                                sen_max = SEN
                # logger.debug("parameter searching accomplished")
                lr = LinearRegression()
                rfe = RFE(lr, n_features_to_select=featureNum_op)
                rfe.fit(X_train_nol_sel, y_train)
                selected2 = rfe.support_
                savemat("Results/selected_{f}/selected2_{c}_{fold}.mat".format(f=featureName, c=iterCount, fold=fold), {'data': selected2})
                clf = svm.SVC(kernel='linear', C=COp).fit(X_train_nol_sel[:, selected2], y_train)
                score = clf.score(X_test_nol_sel[:, selected2], y_test)
                y_score = clf.decision_function(X_test_nol_sel[:, selected2])
                res = clf.predict(X_test_nol_sel[:, selected2])
                ACC, SEN, SPE = model_evaluate(test_label=y_test, res_label=res)
                fpr, tpr, threshold = roc_curve(y_test, y_score)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                logger.debug("ACC={ACC}, SEN={SEN}, SPE={SPE}, AUC={AUC}".format(ACC=ACC, SEN=SEN, SPE=SPE, AUC=roc_auc))
                content = [ACC, SEN, SPE, roc_auc]
                contents.append(content)
                ACCs.append(ACC)
                SENs.append(SEN)
                SPEs.append(SPE)
                AUCs.append(roc_auc)
                fold += 1
        logger.debug("mean: ACC={ACC}±{ACC_std}, SEN={SEN}±{SEN_std}, SPE={SPE}±{SPE_std}, AUC={AUC}±{AUC_std}".format(ACC=sum(ACCs) / len(ACCs), ACC_std=np.std(ACCs),
                                                                                                                       SEN=sum(SENs) / len(SENs), SEN_std=np.std(SENs),
                                                                                                                       SPE=sum(SPEs) / len(SPEs), SPE_std=np.std(SPEs),
                                                                                                                       AUC=sum(AUCs) / len(AUCs), AUC_std=np.std(AUCs)))
        ACC_matrix[k_id, c_id] = sum(ACCs) / len(ACCs)
        AUC_matrix[k_id, c_id] = sum(AUCs) / len(AUCs)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.show()
        if not os.path.exists("Results/ROC_{f}".format(f=featureName)):
            os.mkdir("Results/ROC_{f}".format(f=featureName))
        savemat("Results/ROC_{f}/fpr.mat".format(f=featureName), {'data': mean_fpr})
        savemat("Results/ROC_{f}/tpr.mat".format(f=featureName), {'data': mean_tpr})
        
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
        head = ('ACC', 'SEN', 'SPE', 'AUC')
        for col in range(len(head)):
            sheet.write(0, col, head[col])
        for i in range(len(contents)):
            for col in range(len(contents[i])):
                sheet.write(i + 1, col, contents[i][col])
        workbook.save('Results/{f}.xls'.format(f=featureName))
savemat("Results/ACC_matrix.mat", {"data": ACC_matrix})
savemat("Results/AUC_matrix.mat", {"data": AUC_matrix})