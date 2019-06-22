import numpy as np
import os


class ImageFeatureEncoder:
    """
    Image feature encoder that can be persisted to disk.
    """

    def __init__(self):
        self.shape = None

    def fit(self, X: np.ndarray):
        self.shape = X.shape[1:]

    def transform(self, X: np.ndarray):
        assert X.ndim == 4  # n_examples, img_width, img_height, n_channels
        assert X.shape[3] == 3  # 3 channels
        assert np.array_equal(X.shape[1:], self.shape), f'encoder not tuned for parameters {self.shape}'

        Xnorm = X.astype(np.float16) / 255. * 2.0 - 1.0  # normalize image data between -1 and 1
        return Xnorm

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        target_file = os.path.join(directory, 'image_encoder.npy')
        np.save(target_file, self.shape)

    def load(self, directory):
        target_file = os.path.join(directory, 'image_encoder.npy')
        self.shape = np.load(target_file, self.shape)

