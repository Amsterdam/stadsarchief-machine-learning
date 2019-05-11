import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data import build_ids, load_X, load_Y_yaml
from examples.aanvraag_besluit.load_y import create_Y
from examples.aanvraag_besluit.transformer import Transformer


def getInt(value: str):
    try:
        return int(value)
    except TypeError:
        return None
    except ValueError:
        return None


def load_raw(img_dir, label_dir):
    ids = build_ids(img_dir, label_dir, '.yaml')
    ids = ids[:30]
    print("LIMITING IDS")
    print(f"first few ids: {ids[:5]}")

    X = load_X(img_dir, ids)
    Y = load_Y_yaml(label_dir, ids)
    return [X, Y, ids]


def preprocess_img(X: np.ndarray):
    assert X.ndim == 4
    assert X.shape[3] == 3  # 3 channels
    Xnorm = X / 255.  # normalize image data between 0 and 1
    return Xnorm


def process_attributes(Ymeta: list):
    data = []
    for y in Ymeta:
        # print(y)
        row_data = {}
        row_data['dossier_jaar'] = getInt(y.get('dossier_jaar', None))
        row_data['dossier_type'] = y.get('dossier_type', 'onbekend')
        row_data['stadsdeel_code'] = y.get('stadsdeel_code', 'onbekend')
        # for column in headers:
        #     row_data[column] = y.get(column, None)
        data.append(row_data)
    df = pd.DataFrame(data)
    return df


def preprocess_X(Ximg: np.ndarray, Xdecoded: pd.DataFrame, transformer: Transformer):
    Ximg = preprocess_img(Ximg)

    Xdata = transformer.encode(Xdecoded)
    X = [Ximg, Xdata]

    # print('Xdecoded', Xdecoded[:4])
    # print('Xdata', Xdata[:4])
    XdecodedB = transformer.decode(Xdata)
    # print('XdecodedB', XdecodedB[:4])

    return X


def load_data_aanvraag(set1_dirs, set2_dirs, random_state=42):
    transformer = Transformer()

    print('loading set1')
    [Ximg1, Ymeta1, _] = load_raw(set1_dirs.get('images'), set1_dirs.get('labels'))

    print('loading set2')
    [Ximg2, Ymeta2, _] = load_raw(set2_dirs.get('images'), set2_dirs.get('labels'))

    #
    # Form train, validation and test sets
    # add some from set2 to training
    #
    print('Configuring train, validation and test sets:')
    Ximg_A, Ximg_B, Ymeta_A, Ymeta_B = train_test_split(Ximg2, Ymeta2, test_size=1.0 - 0.33, shuffle=True, random_state=random_state)
    print('adding to train set: ', Ximg_A.shape)
    Ximg_train = np.concatenate([Ximg1, Ximg_A], axis=0)
    Ymeta_train = np.concatenate([Ymeta1, Ymeta_A], axis=0)

    print('remaining for test and validation set: ', Ximg_B.shape)
    Ximg_validation, Ximg_test, Ymeta_valid, Ymeta_test = train_test_split(Ximg_B, Ymeta_B, test_size=0.5, shuffle=True, random_state=random_state)
    # Xvalid = X_validation
    # Yvalid = np.array(y_validation)

    print("not using test set of shape: ", Ximg_test.shape)

    Xdata_train = process_attributes(Ymeta_train)
    Xdata_valid = process_attributes(Ymeta_valid)
    Xdata_test = process_attributes(Ymeta_test)

    Ytrain = create_Y(Ymeta_train, verbose=False)
    Yvalid = create_Y(Ymeta_valid, verbose=False)
    Ytest = create_Y(Ymeta_test, verbose=False)

    Xdata_mix = Xdata_train.append(Xdata_valid)
    transformer.fit(Xdata_mix)

    Xtrain = preprocess_X(Ximg_train, Xdata_train, transformer)
    Xvalid = preprocess_X(Ximg_validation, Xdata_valid, transformer)
    Xtest = preprocess_X(Ximg_test, Xdata_test, transformer)

    return [Xtrain, Ytrain, Xvalid, Yvalid, transformer]
