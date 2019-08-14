import math

import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    """
    Show individual image
    :param image:
    :return:
    """
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def show_prediction_images(
        X: np.ndarray,
        Y: np.ndarray,
        predictions: np.ndarray,
        references: list,
        encoder,
        limit=3,
        filter=None,
        columns=3
):
    """
    Show prediction images along with expected and predicted values
    :param X: images, channel last
    :param Y: One hot true labels
    :param predictions: One hot predictions
    :param references:
    :param encoder:
    :param limit:
    :param filter: list of ids to filter by
    :param columns:
    :return:
    """
    assert (predictions.shape[0] == Y.shape[0])

    confidences = np.max(predictions, axis=1)

    y_class = encoder.inverse_transform(Y)
    pred_class = encoder.inverse_transform(predictions)

    if filter is not None:
        # limit to ids in list
        assert type(filter) == list or type(filter) == np.ndarray
        indices = np.where(np.isin(references, filter))[0]
        indices = indices[:limit]
    else:
        indices = range(0, limit)

    images = X

    count = len(indices)
    plt.figure(figsize=(20, math.ceil(count / columns) * 7))
    print(f'showing {count} image(s)')

    for i, index in enumerate(indices):
        image = images[index, :, :, :]
        plt.subplot(count / columns + 1, columns, i + 1)
        expected = y_class[index]
        confidence = str(round(confidences[index], 2))

        predict_name = pred_class[index]
        id = references[index]
        is_correct = expected == predict_name
        if is_correct:
            plt.gca().set_title(f"{id}, {confidence}: {predict_name}\u2713")
        else:
            plt.gca().set_title(f"{id}, {confidence}: {predict_name} -> {expected}")
        plt.imshow(image)
    plt.show()
