import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


class Transformer(object):

    def __init__(self):
        self.transformers = {}

    def fit(self, df: pd.DataFrame):
        self.encode(df, do_fit=True)

    def encode(self, df: pd.DataFrame, do_fit=False):
        input = df['dossier_jaar'].values
        assert input.ndim == 1
        input = input.reshape(-1, 1)
        assert input.ndim == 2

        if do_fit:
            # None is transformed to NaN
            # Inverse transform from NaN is to NaN
            enc = MinMaxScaler()
            enc.fit(input)
            self.transformers['dossier_jaar'] = enc
        else:
            enc = self.transformers['dossier_jaar']
        dossier_jaar = enc.transform(input)

        input = df['dossier_type']
        # print('input: ', input)
        if do_fit:
            enc = LabelBinarizer()
            # enriched_input = input
            # enriched_input.append(pd.DataFrame({'dossier_type': 'onbekend'}), ignore_index=True)
            # print('enriched_input :', enriched_input)
            enc.fit(input)
            self.transformers['dossier_type'] = enc
        else:
            enc = self.transformers['dossier_type']
        dossier_type = enc.transform(input)

        input = df['stadsdeel_code']

        if do_fit:
            enc = LabelBinarizer()
            enc.fit(input)
            self.transformers['stadsdeel_code'] = enc
        else:
            enc = self.transformers['stadsdeel_code']
        stadsdeel_code = enc.transform(input)

        X = np.hstack([dossier_jaar, dossier_type, stadsdeel_code])
        # print(dossier_jaar[:3])
        # print(dossier_type[:3])
        # print(stadsdeel_code[:3])
        # print(X[:3])
        return X

    def decode(self, data_enc):
        headers = ['dossier_jaar', 'dossier_type', 'stadsdeel_code']

        idx = 0
        enc = self.transformers['dossier_jaar']
        dossier_jaar = data_enc[:, idx, np.newaxis]
        dossier_jaar_dec = enc.inverse_transform(dossier_jaar)
        idx += 1

        enc = self.transformers['dossier_type']
        n = len(enc.classes_)
        dossier_type = data_enc[:, idx:idx+n, np.newaxis]
        dossier_type_dec = enc.inverse_transform(dossier_type)
        idx += n

        enc = self.transformers['stadsdeel_code']
        n = len(enc.classes_)
        stadsdeel_code = data_enc[:, idx:idx+n]
        stadsdeel_code_dec = enc.inverse_transform(stadsdeel_code)
        stadsdeel_code_dec = stadsdeel_code_dec.reshape(-1, 1)
        assert stadsdeel_code_dec.ndim == 2
        idx += 1

        data_np = np.hstack([dossier_jaar_dec, dossier_type_dec, stadsdeel_code_dec])

        df = pd.DataFrame(data_np, columns=headers)
        return df
