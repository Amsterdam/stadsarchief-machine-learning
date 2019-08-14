import numpy as np
import os
from sklearn import preprocessing


class TargetEncoder:

    def __init__(self):
        self.encoder = preprocessing.OneHotEncoder()  # outputs 2d array, multi class classification

    def fit(self, *args, **kwargs):
        return self.encoder.fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.encoder.transform(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs):
        return self.encoder.inverse_transform(*args, **kwargs)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        target_file = os.path.join(directory, 'target_encoder.npy')
        np.save(target_file, self.encoder.categories_)

    def load(self, directory):
        target_file = os.path.join(directory, 'target_encoder.npy')
        categories = np.load(target_file)
        self.encoder.categories_ = categories
        self.encoder._legacy_mode = False  # required so .transform call succeeds. Normally this is set properly when calling .fit
