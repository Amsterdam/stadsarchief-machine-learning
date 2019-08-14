import numpy as np


def split_bool_arrays(predictions: np.ndarray, threshold: float):
    """
    Get boolean array for predictions >= threshold and boolean array for prediction < threshold
    :param predictions: One hot predictions
    :param threshold:
    :return:
    """
    assert predictions.shape[1] == 2, 'expecting binary prediction in one hot format'

    y_pred_conf = np.amax(predictions, axis=1)

    certain = y_pred_conf >= threshold
    uncertain = np.invert(certain)
    return [certain, uncertain]


def split_uncertain(predictions: np.ndarray, threshold: float, elements):
    """
    Split all elements into certain and uncertain buckets

    @return [[elem1_certain, elem1_uncertain], ...]
    """
    for element in elements:
        assert element.shape[0] == predictions.shape[0], 'number of an element not equal to number of predictions'

    [certain, uncertain] = split_bool_arrays(predictions, threshold)

    results = []
    for element in elements:
        certain_bucket = element[certain]
        uncertain_bucket = element[uncertain]
        results.append([certain_bucket, uncertain_bucket])
    return results


def show_threshold_report(Yvalid_raw, Zvalid, predictions, labelEncoder, threshold):
    results = split_uncertain(predictions, threshold, [Img, Data, Label, predictions])
    print('image certain shape: ', results[0][0].shape)
    print('image uncertain shape: ', results[0][1].shape)
    print('meta certain shape: ', results[1][0].shape)
    print('meta uncertain shape: ', results[1][1].shape)
