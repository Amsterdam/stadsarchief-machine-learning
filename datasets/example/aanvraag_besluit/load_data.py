import logging
import math
from itertools import compress

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.data import build_ids, load_X, load_yaml_ids
from ..aanvraag_besluit.load_y import create_Z

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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


def load_raw(img_dir, label_dir, skip: list, limit: int):
    # Build list of all ids that have both an image and associated meta data
    ids = build_ids(img_dir, label_dir, '.yaml', skip)

    if limit:
        ids = ids[:limit]
        print(f"Limiting to {limit}")
    print(f"first few ids: {ids[:5]}")
    print(f"last few ids: {ids[-5:]}")

    # Remove unlabeled ids
    yaml = load_yaml_ids(label_dir, ids)
    print(f'ids count: {len(ids)}')
    ids = filter_unlabeled(yaml, ids)
    print(f'ids with label count: {len(ids)}')
    print('loading yaml filtered')
    yaml = load_yaml_ids(label_dir, ids)

    log.info('loading images')
    X = load_X(img_dir, ids)
    print(f'type of nparray: {X.dtype}')
    print(f'{round(X.nbytes / 1024**2, 3)}MB')

    log.info('processing meta data')
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
    return df.values  # We return it as a numpy array so to avoid trouble in a later stage


def load_set(multiple_inputs, skip: list):
    Img_acc = None
    Data_acc = None
    Label_acc = None
    for input_dirs in multiple_inputs:
        print('--- loading set: ', input_dirs)
        limit = input_dirs.get('limit')
        print("Limit:", limit)
        [Img, Data, Label, _] = load_raw(input_dirs.get('images'), input_dirs.get('labels'), skip, limit=limit)
        print(f'unique counts: {np.unique(Label, return_counts=True)}')
        print(f'Img shape: {Img.shape}')
        if Img_acc is None:
            Img_acc = Img
            Data_acc = Data
            Label_acc = Label
        else:
            Img_acc = np.vstack((Img_acc, Img))
            Data_acc = np.vstack((Data_acc, Data))
            Label_acc = np.vstack((Label_acc, Label))
    return [Img_acc, Data_acc, Label_acc]


def load_data_aanvraag(img_dim, random_state=42):
    """
    Load train and test set input and split into train, dev and hold out test set
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

    # Represents actual problem space
    inputs = [
        {
            'images': f'examples/aanvraag_besluit/dataset_3b_ZO_AnB_other_production/images/{img_dim[0]}x{img_dim[1]}/',
            'labels': 'examples/aanvraag_besluit/dataset_3b_ZO_AnB_other_production/labels/',
            'limit': 2364
        },
        {
            'images': f'examples/aanvraag_besluit/dataset_4_ZO_other_production/images/{img_dim[0]}x{img_dim[1]}/',
            'labels': 'examples/aanvraag_besluit/dataset_4_ZO_other_production/labels/',
            'limit': 332
        },
    ]

    # May be larger than problem space (contain synthetic images or tangentially related)
    inputs_train_only = [
        {
            'images': f'examples/aanvraag_besluit/dataset_3a_ZO_AnB_aanvragen/images/{img_dim[0]}x{img_dim[1]}/',
            'labels': 'examples/aanvraag_besluit/dataset_3a_ZO_AnB_aanvragen/labels/',
            'limit': 700
        },
        {
            'images': f'examples/aanvraag_besluit/dataset_1_mixed_hand_annotated/resized/{img_dim[0]}x{img_dim[1]}/',
            'labels': 'examples/aanvraag_besluit/dataset_1_mixed_hand_annotated/labels/'
        },
        {
            'images': f'examples/aanvraag_besluit/dataset_2_oost_hand_annotated/images/{img_dim[0]}x{img_dim[1]}/',
            'labels': 'examples/aanvraag_besluit/dataset_2_oost_hand_annotated/labels/'
        }
    ]

    ids_to_skip = [
        # Images are not loaded properly
        'ST00122908_00001',
        'ST00058502_00001'
    ]

    #
    # Stage 1, load features and labels for both input sets
    #
    [Img_in, Data_in, Label_in] = load_set(inputs, ids_to_skip)
    print('Img_in.shape', Img_in.shape)
    print('Data_in.shape', Data_in.shape)
    print('Label_in.shape', Label_in.shape)
    print()

    [Img_in_train, Data_in_train, Label_in_train] = load_set(inputs_train_only, ids_to_skip)
    print('Img_in_train.shape', Img_in_train.shape)
    print('Data_in_train.shape', Data_in_train.shape)
    print('Label_in_train.shape', Label_in_train.shape)
    print('Img_in_train.shape', Img_in_train.shape)
    print()

    #
    # Stage 2, redistribute data to form Train, validation and test sets
    #
    # Train = inputs_train + some examples from inputs set
    # Validation = subset of inputs set
    # (hold out) Test = subset of inputs set
    count = Img_in.shape[0]
    splits = [int(.55 * count), int(.99 * count)]
    print('splits', splits)
    [Img_train_extra, Img_valid, Img_test] = np.vsplit(Img_in, splits)
    [Data_train_extra, Data_valid, Data_test] = np.vsplit(Data_in, splits)
    [Label_train_extra, Label_valid, Label_test] = np.vsplit(Label_in, splits)

    Img_train = np.vstack((Img_in_train, Img_train_extra))
    Data_train = np.vstack((Data_in_train, Data_train_extra))
    Label_train = np.vstack((Label_in_train, Label_train_extra))

    # Shuffle everything
    Img_train, Data_train, Label_train = shuffle(Img_train, Data_train, Label_train, random_state=random_state)
    Img_valid, Data_valid, Label_valid = shuffle(Img_valid, Data_valid, Label_valid, random_state=random_state)
    Img_test, Data_test, Label_test = shuffle(Img_test, Data_test, Label_test, random_state=random_state)

    return [
        [Img_train, Data_train, Label_train],
        [Img_valid, Data_valid, Label_valid],
        [Img_test, Data_test, Label_test],
    ]


def load_getting_started_data(img_dim, random_state=42):
    """
    Load train, dev and hold out test set for getting started dataset
    :return:
    """

    # Represents actual problem space
    inputs = [
        {
            'images': f'datasets/example/aanvraag_besluit/dataset_0/images/{img_dim[0]}x{img_dim[1]}/',
            'labels': 'datasets/example/aanvraag_besluit/dataset_0/labels/'
        }
    ]

    ids_to_skip = []
    splits = [.6, .8]  # Train 60%, Dev/validation 20% and 20% test set

    #
    # Stage 1, load features and labels for input set
    #
    [Img_in, Data_in, Label_in] = load_set(inputs, ids_to_skip)
    print('Img_in.shape', Img_in.shape)
    print('Data_in.shape', Data_in.shape)
    print('Label_in.shape', Label_in.shape)
    print()

    #
    # Stage 2, redistribute data to form Train, validation and test sets
    #
    count = Img_in.shape[0]
    splits_indices = [int(splits[0] * count), int(splits[1] * count)]
    print('splits', splits)
    [Img_train, Img_valid, Img_test] = np.vsplit(Img_in, splits_indices)
    [Data_train, Data_valid, Data_test] = np.vsplit(Data_in, splits_indices)
    [Label_train, Label_valid, Label_test] = np.vsplit(Label_in, splits_indices)

    # Shuffle everything
    Img_train, Data_train, Label_train = shuffle(Img_train, Data_train, Label_train, random_state=random_state)
    Img_valid, Data_valid, Label_valid = shuffle(Img_valid, Data_valid, Label_valid, random_state=random_state)
    Img_test, Data_test, Label_test = shuffle(Img_test, Data_test, Label_test, random_state=random_state)

    return [
        [Img_train, Data_train, Label_train],
        [Img_valid, Data_valid, Label_valid],
        [Img_test, Data_test, Label_test],
    ]
