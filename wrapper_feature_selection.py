import os
import math

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, confusion_matrix
from scipy import stats
from statsmodels.stats import multitest

import pycasso


class WrapperFeatureSelection():  # lasso, mcp, scad
    def __init__(self, lambda_val=1, mask_save_path=None):
        self._lambda_val = lambda_val
        self._mask_save_path = mask_save_path
        return

    def get_lasso_mask(self, X, y, method='l1', family="binomial"):  # # ("l1", "mcp", "scad")
        X = np.array(X)
        y = np.array(y)

        data_shape = X.shape

        # flatten data
        if len(data_shape) > 2:
            _X = np.array([elem.flatten() for elem in X])
            print("_X.shape", _X.shape)
        else:
            _X = X

        lambda_max = np.max(
            np.abs(np.matmul(_X.T, y))) / data_shape[0]

        print("lambda_max", lambda_max)
        print("min_lambda_ratio", (1 / lambda_max * self._lambda_val))
        solver_l1 = pycasso.Solver(_X, y, lambdas=(2, (1 / lambda_max * self._lambda_val)), family=family,
                                   penalty=method)
        solver_l1.train()

        # ## lambdas used
        # ## Obtain the result
        result = solver_l1.coef()
        # print(solver_l1.lambdas)

        i = 1
        betas = result['beta'][i]
        mask_map = np.zeros(betas.shape)
        mask_map[betas == 0] = 0
        mask_map[betas != 0] = 1

        # save_selected_features
        if self._mask_save_path is not None:
            df = pd.DataFrame({'%s_lambda_%f' % (method, self._lambda_val): betas,
                               '%s_mask_%f' % (method, self._lambda_val): mask_map})
            df.to_excel(self._mask_save_path)

        return mask_map.reshape(data_shape[1:])


class MarginalTest():
    def __init__(self, target_path=None, dtype="img"):
        self._samples = None
        self._labels = None
        return

    def updateData(self, datas, labels):
        self._samples = datas
        self._labels = labels
        return

    def create_p_matrix(self, x1, x2, type_test='anova'):
        """

        :param x1: (N, F) or (N, X, Y)
        :param x2: (N, F) or (N, X, Y)
        :param type_test:
        :return: statistics like (F,) or (X, Y)
        """
        p_matrix = None
        if type_test == 'anova':
            print("[!] create p-value matrix with ANOVA")
        elif type_test == 'ttest_ind':
            print("[!] create p-value matrix with ttest_ind")
            statstics = stats.ttest_ind(x1, x2, axis=0)
        elif type_test == 'paired_t_test':
            print("[!] create p-value matrix with paired_t_test")

        return statstics

    def fdr_masking(self, p_vals, alpha=0.05, method='fdr_bh'):
        """
        :param p_vals:
        :param alpha:
        :param method:
        :return: (reject, pvals_corrected)
        """

        p_vals_flat = p_vals.flatten()
        res = multitest.multipletests(pvals=p_vals_flat, alpha=alpha, method=method, is_sorted=False,
                                      returnsorted=False)

        return res[0].reshape(p_vals.shape), res[1].reshape(p_vals.shape)

    def get_mask(self, X_datas, y_labels, type_test='ttest_ind', method='fdr_bh'):
        if len(y_labels) == 2:
            tr_normal = X_datas[y_labels == 0]
            tr_abnormal = X_datas[y_labels == 1]

            statistics = self.create_p_matrix(tr_normal, tr_abnormal, type_test=type_test)
            # adjust p-values
            statistics = self.fdr_masking(statistics[1], alpha=0.0001, method=method)
            adjust_pval_map = statistics[1]
            selected_mask = statistics[0]

        return selected_mask


def get_performances(target, output, pos_label=1):
    # Evaluation 1 : Accuracy
    _acc = accuracy_score(target, output)

    # Evaluation 2 : AUROC
    fpr, tpr, _ = roc_curve(target, output, pos_label=pos_label)
    _auroc = auc(fpr, tpr)

    # Evaluation 3 : F1-score
    _f1_score = f1_score(target, output)

    # Evaluation 4 : geometric mean
    conf_mat = confusion_matrix(target, output)
    specificity = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    sensitivity = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
    _g_mean = stats.gmean([specificity, sensitivity])

    return _acc, _auroc, _f1_score, _g_mean


# preprocess 2 : grid expression
def express_on_grid(input_data):
    input_data = np.array(input_data)
    data_shape = input_data.shape
    grid_rlen = math.ceil(math.sqrt(data_shape[1]))
    if grid_rlen * (grid_rlen - 1) > data_shape[1]:
        grid_clen = grid_rlen - 1
    else:
        grid_clen = grid_rlen

    zero_pads = np.zeros([data_shape[0], grid_rlen * grid_clen - data_shape[1]])
    concat_data = np.concatenate([input_data, zero_pads], axis=1)
    reshaped_data = concat_data.reshape(data_shape[0], grid_rlen, grid_clen)
    return reshaped_data
