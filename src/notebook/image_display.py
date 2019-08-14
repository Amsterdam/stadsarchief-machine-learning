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
        predictions: list,
        references: list,
        encoder,
        limit=3,
        columns=3
):
    assert (predictions.shape[0] == Y.shape[0])
    # if len(Y.shape) != 1:
    #     # Not binary classification
    #     Y = np.argmax(Y, axis=1)
    #     Y = Y.reshape(-1, 1)

    predictions_id = np.argmax(predictions, axis=1)

    y_class = encoder.inverse_transform(Y)
    pred_class = encoder.inverse_transform(predictions)

    plt.figure(figsize=(20, math.ceil(limit / columns) * 7))

    images = X[:limit, :, :, :]
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        expected = y_class[i]
        predict_confidence = predictions[i, predictions_id[i]]
        rounded = str(round(predict_confidence, 2))

        predict_name = pred_class[i]
        id = references[i]
        is_correct = expected == predict_name
        if is_correct:
            plt.gca().set_title(f"{id}, {rounded}: {predict_name}\u2713")
        else:
            plt.gca().set_title(f"{id}, {rounded}: {predict_name} -> {expected}")
        plt.imshow(image)
    plt.show()
