import math

import numpy as np
import matplotlib.pyplot as plt


def convert_to_index(Y, types):
    return np.array([types.index(y) for y in Y])


def list_stats(Y):
    types = list(set(Y))
    print(f"classes: {len(types)}")

    Y_idx = convert_to_index(Y, types)
    unique, counts = np.unique(Y_idx, return_counts=True)
    unique_counts = list(zip(unique, counts))
    #     print(unique_counts)

    min_cnt = 5
    print(f"classes with count less than {min_cnt} ignored")
    top = [[types[idx], count] for idx, count in unique_counts if count > min_cnt]
    print(f"count classes: {top}")

    largest_class = max(unique_counts, key=lambda x: x[1])
    print(f"largest class: {types[largest_class[0]]}, count: {largest_class[1]}")

    total_count = Y.shape[0]
    print(f"total count: {total_count}")

    print(f"score to beat: {largest_class[1] / total_count}")


def show_train_curves(history):
    """
    based on: https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
    """
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)


def show_prediction_list(predictions, expected, show_cnt = 30):
    if len(expected.shape) != 1:
        # Not binary classification
        expected = np.argmax(expected, axis=1)

    predictions_classes = np.argmax(predictions, axis=1)
    print(f"index:      [{' '.join(str(x)[-1:] for x in range(show_cnt))}]")
    print(f"prediction: {predictions_classes[:show_cnt]}")
    print(f"expected:   {expected[:show_cnt]}")


def show_prediction_images(X, Y, ids, predictions, types, limit=3, columns=3):
    if len(Y.shape) != 1:
        # Not binary classification
        Y = np.argmax(Y, axis=1)

    predictions_classes = np.argmax(predictions, axis=1)

    assert(predictions_classes.shape[0] == Y.shape[0])
    plt.figure(figsize=(20, math.ceil(limit/columns) * 7) )

    images = X[:limit, :, :, :]
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        expected = types[Y[i]]
        predict_confidence = predictions[i, predictions_classes[i]]
        rounded = str(round(predict_confidence, 2))

        predict_name = types[predictions_classes[i]]
        id = ids[i]
        is_correct = expected == predict_name
        if is_correct:
            plt.gca().set_title(f"{id}, {rounded}: {predict_name}\u2713")
        else:
            plt.gca().set_title(f"{id}, {rounded}: {predict_name} -> {expected}")
        plt.imshow(image)
    plt.show()


def show_prediction_images_new(X, Y, predictions, meta, encoder, limit=3, columns=3):
    if len(Y.shape) != 1:
        # Not binary classification
        Y = np.argmax(Y, axis=1)

    # print(encoder.classes_)
    # print(predictions[:10])

    # binary classes floating prediction to class prediction
    predictions_classes = predictions >= 0.5 * 1.0
    predictions_classes = predictions_classes.astype(np.int)

    # print(predictions_classes[:10])
    # print(encoder.inverse_transform(predictions_classes)[:10])

    # predictions_classes = np.argmax(predictions, axis=1)

    assert(predictions_classes.shape[0] == Y.shape[0])

    plt.figure(figsize=(20, math.ceil(limit/columns) * 7))

    images = X[:limit, :, :, :]
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        expect_class = Y[i]
        expect_name = encoder.inverse_transform([expect_class])
        predict_class = predictions_classes[i][0]
        predict_name = encoder.inverse_transform([predict_class])
        # print(expect_class)
        # print(predict_class)
        # print('---')

        predict_confidence = predictions[i][0]
        rounded = str(round(predict_confidence, 2))

        id = meta[i].get('reference')
        is_correct = predict_class == expect_class
        if is_correct:
            plt.gca().set_title(f"{id}, {rounded}: {predict_name}\u2713")
        else:
            plt.gca().set_title(f"{id}, {rounded}: {predict_name} -> {expect_name}")
        plt.imshow(image)
    plt.show()

