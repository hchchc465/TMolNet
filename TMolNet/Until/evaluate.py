import os
import torch

import numpy as np

from math import sqrt
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error

def eval_rocauc(input_dict):
    '''
        compute ROC-AUC averaged across tasks
    '''
    y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            y = np.array(y_true[:, i], dtype=float)
            y[y < 0] = np.nan
            is_labeled = y == y
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}


# 在 eval_rmse 函数中添加调试信息
import numpy as np


import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def eval_rmse(input_dict):

    y_true = np.asarray(input_dict['y_true'])
    y_pred = np.asarray(input_dict['y_pred'])

    # 计算 RMSE
    rmse_value = sqrt(mean_squared_error(y_true, y_pred))
    return {'rmse': rmse_value}
