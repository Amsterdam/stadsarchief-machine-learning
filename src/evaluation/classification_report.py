import logging
import numpy as np

from sklearn.metrics import classification_report

log = logging.getLogger(__name__)


def _print_missing_types_error(unique_types: list, expected: list, name: str):
    log.warning(f'{name} samples do not reflect all types, recall, precision and T1 scores therefor do not make much sense.')
    log.warning(f'{name} types: {unique_types}, expected {expected}')
    log.warning('Classification report generation skipped.')


def create_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: list
):
    assert y_true.shape[1] == 1, 'expecting column vector (not one hot encoded vector)'
    assert y_true.shape == y_pred.shape

    true_types = np.unique(y_true)
    pred_types = np.unique(y_pred)

    report = None
    if len(true_types) < len(categories):
        _print_missing_types_error(true_types, categories, 'True')
    elif len(pred_types) < len(categories):
        _print_missing_types_error(pred_types, categories, 'Pred')
    else:
        report = classification_report(y_true, y_pred, labels=categories)
    return report
