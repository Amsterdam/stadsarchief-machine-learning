import numpy as np
import pandas as pd


def create_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
):
    assert y_true.shape[1] == 1, 'expecting column vector (not one hot encoded vector)'
    assert y_true.shape == y_pred.shape

    y_true_pd = pd.Series(y_true.ravel())
    y_pred_pd = pd.Series(y_pred.ravel())
    crosstab = pd.crosstab(y_true_pd, y_pred_pd, rownames=['Actual'], colnames=['Predicted'], margins=True, margins_name='Total')

    return crosstab
