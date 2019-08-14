import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from src.notebook.image_display import show_prediction_images
from src.evaluation import confusion_matrix, classification_report
from src.evaluation.split_report import split_dataframe
from src.predict.threshold import split_uncertain


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def show_train_curves(history):
    """
    based on: https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
    """
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)


def combined_report(
        true_oh: np.ndarray,
        predictions_oh: np.ndarray,
        ids: np.ndarray,
        label_encoder,
        threshold: float,
):
    assert true_oh.shape[1] == 2, 'expecting binary one hot inputs'
    assert true_oh.shape == predictions_oh.shape

    categories = ['aanvraag', 'other']

    true = label_encoder.inverse_transform(true_oh)
    pred = label_encoder.inverse_transform(predictions_oh)
    is_correct = true == pred

    results = split_uncertain(predictions_oh, threshold, [true, pred, is_correct, ids])

    [
        true_splits,
        pred_splits,
        is_correct_splits,
        ids_splits
    ] = results

    certain_count = true_splits[0].shape[0]
    uncertain_count = true_splits[1].shape[0]
    total = certain_count + uncertain_count
    certain_percentage = certain_count / total * 100
    uncertain_percentage = uncertain_count / total * 100

    counts_df = split_dataframe(certain_count, uncertain_count)

    if certain_count > 0:
        true_labels = true_splits[0]
        pred_labels = pred_splits[0]
        certain_accuracy = accuracy_score(true_labels, pred_labels)
        certain_recall = recall_score(true_labels, pred_labels, pos_label='aanvraag')
        print(f'certain examples:\t\t{Color.BOLD}{round(certain_percentage, 1)}%{Color.END}', end='')
        print(f'\t accuracy: {Color.BOLD}{round(certain_accuracy*100, 2)}%{Color.END}', end='')
        print(f', aanvraag recall: {Color.BOLD}{round(certain_recall*100, 2)}%{Color.END}')

    else:
        print(f'certain examples:\t\t{Color.BOLD}{round(certain_percentage, 1)}%{Color.END}')
    print(f'uncertain examples:\t\t{Color.BOLD}{round(uncertain_percentage, 1)}%{Color.END}')


    print()
    print()
    print(counts_df.round(2))

    print()
    print()
    print(f'{Color.BOLD}## Certain examples{Color.END}')
    if certain_count > 0:
        matrix = confusion_matrix.create_matrix(true_splits[0], pred_splits[0])
        print(matrix)
        report = classification_report.create_report(true_splits[0], pred_splits[0], categories)
        print(report)
    else:
        print('no data')

    print()
    print()
    print(f'{Color.BOLD}## Uncertain examples{Color.END}')
    if uncertain_count > 0:
        matrix = confusion_matrix.create_matrix(true_splits[1], pred_splits[1])
        print(matrix)
        report = classification_report.create_report(true_splits[1], pred_splits[1], categories)
        print(report)
    else:
        print('no data')

    print()
    print()
    print(f'{Color.BOLD}## Certain errors{Color.END}')

    # In the "certain" predictions, get the incorrect predictions (false positive and false negatives)
    certain_ids = ids_splits[0]
    certain_ids.shape = (certain_ids.size, 1)
    certain_is_correct = is_correct_splits[0]
    certain_is_incorrect = np.invert(certain_is_correct)

    incorrect_ids = certain_ids[certain_is_incorrect]


    show_max = 50
    print(f'found {len(incorrect_ids)} incorrect examples')
    print(f'incorrect ids[:{show_max}]:')
    print('\n'.join(incorrect_ids[:show_max]))

    return incorrect_ids
