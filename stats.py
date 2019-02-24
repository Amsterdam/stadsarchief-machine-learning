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
    top = [[types[idx], count] for idx, count in unique_counts if count > min_cnt]
    print(f"> {min_cnt} count classes: {top}")

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
