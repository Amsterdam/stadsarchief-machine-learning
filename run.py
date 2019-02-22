import numpy as np
from PIL import Image
from scipy import misc
from data import load_data

IMG_DIR = 'examples/0-src/200x200/';
LABEL_DIR = 'examples/0-src/labels/';

[X, Y] = load_data(IMG_DIR, LABEL_DIR)
X.shape
