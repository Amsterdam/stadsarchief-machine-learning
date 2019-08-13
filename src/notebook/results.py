from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

from helper import split_uncertain


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


def show_results(threshold, Yvalid_raw, Zvalid, predictions, labelEncoder):
    ids = Yvalid_raw[:, 2]
    total = predictions.shape[0]
    [certain, uncertain] = split_bool_arrays(predictions, threshold)
    certain_count = np.sum(certain)
    uncertain_count = np.sum(uncertain)

    # Show counts of split
    certain_percentage = certain_count/total*100
    uncertain_percentage = uncertain_count/total*100
    counts_df = pd.DataFrame([
        [certain_count, uncertain_count, total],
        [certain_percentage, uncertain_percentage, 100.0]
    ],
        columns=['certain', 'uncertain', 'total'],
        index=['absolute', 'relative'])

    # Show metrics of splits
    [
        Ytrue_buckets,
        Ypred_buckets,
        ids_buckets,
    ] = split_uncertain(predictions, threshold, [Zvalid, predictions, ids])


    certain_not_empty = Ytrue_buckets[0].shape[0] > 0
    if certain_not_empty:
        y_true = labelEncoder.inverse_transform(Ytrue_buckets[0])
        y_pred = labelEncoder.inverse_transform(Ypred_buckets[0])
        certain_accuracy = accuracy_score(y_true, y_pred)
        certain_recall = recall_score(y_true, y_pred, pos_label='aanvraag')
        print(f'certain examples:\t\t{Color.BOLD}{round(certain_percentage, 1)}%{Color.END}', end='')
        print(f'\t accuracy: {Color.BOLD}{round(certain_accuracy*100, 2)}%{Color.END}', end='')
        print(f', aanvraag recall: {Color.BOLD}{round(certain_recall*100, 2)}%{color.END}')

    else:
        print(f'certain examples:\t\t{color.BOLD}{round(certain_percentage, 1)}%{color.END}')
    print(f'uncertain examples:\t\t{color.BOLD}{round(uncertain_percentage, 1)}%{color.END}')

    print()
    print()
    print(counts_df.round(2))

    print()
    print()
    print(f'{color.BOLD}## Certain examples{color.END}')
    if certain_not_empty:
        certain_is_correct = show_reports(Ytrue_buckets[0], Ypred_buckets[0])
    else:
        print('no data')

    print()
    print()
    print(f'{color.BOLD}## Uncertain examples{color.END}')
    if Ytrue_buckets[1].shape[0] == 0:
        print('no data')
    else:
        uncertain_is_correct = show_reports(Ytrue_buckets[1], Ypred_buckets[1])

    print()
    print()
    print(f'{color.BOLD}## Certain errors{color.END}')
    certain_ids = ids_buckets[0]
    certain_ids.shape = (certain_ids.size, 1)
    certain_is_incorrect = np.invert(certain_is_correct)

    show_max = 50
    incorrect = certain_ids[certain_is_incorrect]
    print(f'incorrect ids[:{show_max}]:')
    print('\n'.join(incorrect[:show_max]))
