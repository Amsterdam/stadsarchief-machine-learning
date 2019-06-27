from itertools import compress

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.data import build_ids, load_X, load_yaml_ids
from examples.aanvraag_besluit.load_y import create_Z


def getInt(value: str):
    try:
        return int(value)
    except TypeError:
        return None
    except ValueError:
        return None


def filter_unlabeled(yaml, data_list: list):
    bool_arr = [item.get('type') != '' for item in yaml]
    data_list_filtered = list(compress(data_list, bool_arr))
    return data_list_filtered


def load_raw(img_dir, label_dir):
    # Build list of all ids that have both an image and associated meta data
    ids = build_ids(img_dir, label_dir, '.yaml')
    ids = ids[:2000]
    print("LIMITING IDS !!!")
    print(f"first few ids: {ids[:5]}")

    # Remove unlabeled ids
    yaml = load_yaml_ids(label_dir, ids)
    print(f'ids count: {len(ids)}')
    ids = filter_unlabeled(yaml, ids)
    yaml = load_yaml_ids(label_dir, ids)
    print(f'ids with label count: {len(ids)}')

    X = load_X(img_dir, ids)
    Y = process_attributes(yaml)
    Z = create_Z(yaml, verbose=False)
    Z = Z.reshape(-1, 1)  # numpy vector (dim 1) to numpy ndarray matrix of dim 2
    return [X, Y, Z, ids]


def process_attributes(Ymeta: list):
    data = []
    for y in Ymeta:
        row_data = {}
        row_data['dossier_jaar'] = getInt(y.get('dossier_jaar', 2050.0))
        row_data['dossier_type'] = y.get('dossier_type', 'onbekend')
        row_data['stadsdeel_code'] = y.get('stadsdeel_code', 'onbekend')
        row_data['reference'] = y.get('reference')
        # for column in headers:
        #     row_data[column] = y.get(column, None)
        data.append(row_data)
    df = pd.DataFrame(data)
    return df


def load_set(multiple_inputs):
    Img_acc = None
    Data_acc = None
    Label_acc = None
    for input_dirs in multiple_inputs:
        print('--- loading set: ', input_dirs)
        [Img, Data, Label, _] = load_raw(input_dirs.get('images'), input_dirs.get('labels'))
        print(f'shape this set Img: {Img.shape}')
        if Img_acc is None:
            Img_acc = Img
            Data_acc = Data
            Label_acc = Label
        else:
            # Xacc[0] = np.vstack((Xacc[0], X[0]))
            # Xacc[1] = np.vstack((Xacc[1], X[1]))
            Img_acc = np.vstack((Img_acc, Img))
            Data_acc = np.vstack((Data_acc, Data))
            Label_acc = np.vstack((Label_acc, Label))
    return [Img_acc, Data_acc, Label_acc]


def load_data_aanvraag(inputs, inputs_train_only, random_state=42):
    """
    Load train and test set input and split into train, dev and hold out test set
    :param inputs: Represents actual problem space
    :param inputs_train_only: May be larger than problem space (contain synthetic images or tangentially related)
    :param random_state:
    :return:
    """

    #
    #  Input dataset                     Splits
    #
    #
    # +-----------+                    +-------------+
    # |           |                    |             |
    # |           |                    |             |
    # |           |                    |    Train    |
    # |           |                    |   100% A    |
    # |  Train    |                    |             |
    # |    A      |                    |             |
    # |           |                    +-------------+
    # |           |
    # |           |
    # |           |                    +-------------+
    # +-----------+                    |             |
    #                    =======>      |             |
    # +-----------+                    |  Validation |
    # |           |                    |             |
    # |           |                    |             |
    # |           |                    |             |
    # |           |                    +-------------+
    # | Represent-|
    # |   ative   |                    +-------------+
    # |    B      |                    |             |
    # |           |                    |             |
    # |           |                    |    Test     |
    # |           |                    |             |
    # |           |                    |             |
    # +-----------+                    +-------------+
    #

    # Stage 1, load features and labels for both input sets
    [Img_in, Data_in, Label_in] = load_set(inputs)
    print('Img_in.shape', Img_in.shape)
    print('Data_in.shape', Data_in.shape)
    print('Label_in.shape', Label_in.shape)

    [Img_in_train, Data_in_train, Label_in_train] = load_set(inputs_train_only)
    print('Img_in_train.shape', Img_in_train.shape)
    print('Data_in.shape', Data_in_train.shape)
    print('Label_in_train.shape', Label_in_train.shape)

    # Stage 2, redistribute data to form Train, validation and test sets

    # Train = inputs_train + some examples from inputs set
    # Validation = subset of inputs set
    # (hold out) Test = subset of inputs set
    count = Img_in.shape[0]
    splits = [int(.33 * count), int(.66*count)]
    print('splits', splits)
    [Img_train_extra, Img_valid, Img_test] = np.vsplit(Img_in, splits)
    [Data_train_extra, Data_valid, Data_test] = np.vsplit(Data_in, splits)
    [Label_train_extra, Label_valid, Label_test] = np.vsplit(Label_in, splits)
    print('Img_train_extra.shape', Img_train_extra.shape)

    Img_train = np.vstack((Img_in_train, Img_train_extra))
    Data_train = np.vstack((Data_in_train, Data_train_extra))
    Label_train = np.vstack((Label_in_train, Label_train_extra))

    return [
        [Img_train, Data_train, Label_train],
        [Img_valid, Data_valid, Label_valid],
        [Img_test, Data_test, Label_test],
    ]
