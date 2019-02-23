import glob
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


def get_label_path(label_dir, id):
    filename = f"{id}.yaml"
    return os.path.join(label_dir, filename)


def load_X(img_dir, ids):
    X = []
    for id in ids:
        path = get_image_path(img_dir, id)
        X.append(np.array(Image.open(path)))
    return np.array(X)


def load_Y(label_dir, ids):
    Y = []
    for id in ids:
        path = get_label_path(label_dir, id)
        Y.append(load_yaml(path))
    return Y


def build_ids(img_dir, label_dir):
    ids = []

    file_paths = glob.glob(f"{img_dir}/*.jpg")

    if len(file_paths) == 0:
        print(f'no images found, {img_dir} exists?')

    for fname in file_paths:
        basename = os.path.basename(fname)
        id, _ = os.path.splitext(basename)
        yaml_path = get_label_path(label_dir, id)
        if os.path.isfile(yaml_path):
            ids.append(id)
        else:
            print(f"missing label for id: {id}")

    return ids


def load_data(img_dir, label_dir):
    ids = build_ids(img_dir, label_dir)

    print(f"first few ids: {ids[:5]}")

    X = load_X(img_dir, ids)
    Y = load_Y(label_dir, ids)


    return [X, Y]

