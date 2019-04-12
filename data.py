import glob
import math
import os

import numpy as np
import yaml
from PIL import Image


def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def get_image_path(img_dir, id):
    filename = f"{id}.jpg"
    return os.path.join(img_dir, filename)


def get_label_path(label_dir, id, extension):
    filename = f"{id}{extension}"
    return os.path.join(label_dir, filename)


def load_X(img_dir, ids):
    X = []
    for id in ids:
        path = get_image_path(img_dir, id)
        X.append(np.array(Image.open(path)))
    return np.array(X)


def load_Y_yaml(label_dir, ids):
    Y = []
    for id in ids:
        path = get_label_path(label_dir, id, '.yaml')
        Y.append(load_yaml(path))
    return Y


def build_ids(img_dir, label_dir, label_extension):
    ids = []

    file_paths = glob.glob(f"{img_dir}/*.jpg")

    if len(file_paths) == 0:
        print(f'no images found, {img_dir} exists?')

    for fname in file_paths:
        basename = os.path.basename(fname)
        id, _ = os.path.splitext(basename)
        label_path = get_label_path(label_dir, id, extension=label_extension)
        if os.path.isfile(label_path):
            if os.stat(label_path).st_size == 0:
                print(f"label is empty: {id}")
            else:
                ids.append(id)
        else:
            print(f"missing label for id: {id}")

    return ids


def load_data(img_dir, label_dir):
    ids = build_ids(img_dir, label_dir, '.yaml')

    print(f"first few ids: {ids[:5]}")

    X = load_X(img_dir, ids)
    Y = load_Y_yaml(label_dir, ids)

    return [X, Y, ids]


def split_data(X, Y, ids, split):
    """
    Split two numpy arrays into two based on split percentage
    :param X: input numpy array
    :param Y: output numpy array
    :param split: fraction of total to apply split.
    :return:
    """
    assert(0.0 < split < 1.0)
    total = X.shape[0]
    assert(total > 0)
    assert(X.shape[0] == Y.shape[0])
    N = math.floor(split * total)
    X_train = X[:N]
    Y_train = Y[:N]
    ids_train =  ids[:N]
    X_test = X[N:]
    Y_test = Y[N:]
    ids_test = ids[N:]
    return (X_train, Y_train, ids_train), (X_test, Y_test, ids_test)
