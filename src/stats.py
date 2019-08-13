import logging

import numpy as np


log = logging.getLogger(__name__)


def _convert_to_index(Y, types):
    return np.array([types.index(y) for y in Y])


def list_label_stats(labels: list):
    """
    Log stats on categories in list
    :param labels:
    :return:
    """
    total_count = labels.shape[0]
    types = list(set(labels))
    log.info(f'classes: {len(types)}')

    indices = _convert_to_index(labels, types)
    unique, counts = np.unique(indices, return_counts=True)
    unique_counts = list(zip(unique, counts))
    #     log.info(unique_counts)

    top = [[types[idx], count] for idx, count in unique_counts]
    log.info(f'count classes: {top}')

    largest_class = max(unique_counts, key=lambda x: x[1])
    log.info(f'largest class: {types[largest_class[0]]}, count: {largest_class[1]}')

    log.info(f'total count: {total_count}')

    log.info(f'score to beat: {largest_class[1] / total_count}')


def show_prediction_list(predictions: np.ndarray, expected: np.ndarray, show_cnt=30):
    """
    List prediction vs expectation for first show_cnt examples
    :param predictions:
    :param expected:
    :param show_cnt:
    :return:
    """
    if len(expected.shape) != 1:
        # Not binary classification
        expected = np.argmax(expected, axis=1)

    predictions_classes = np.argmax(predictions, axis=1)
    log.info(f"index:      [{' '.join(str(x)[-1:] for x in range(show_cnt))}]")
    log.info(f"prediction: {predictions_classes[:show_cnt]}")
    log.info(f"expected:   {expected[:show_cnt]}")
