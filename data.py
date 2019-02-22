import glob
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


def load_data(img_dir, label_dir):
    if img_dir:
        file_paths = glob.glob(f"{img_dir}/*.jpg")
        print(f"first few files: {file_paths[:5]}")
        X = np.array([np.array(Image.open(fname)) for fname in file_paths])
    else:
        X = np.array([])

    # TODO: make sure file line up betweeen img and label dir
    file_paths = glob.glob(f"{label_dir}/*.yaml")
    print(f"first labels   : {file_paths[:5]}")
    Y = np.array([load_yaml(fname) for fname in file_paths])
    return [X, Y]

