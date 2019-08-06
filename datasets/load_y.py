import numpy as np


from .transform import transform_aanvraag_labels


def extract_type(Ymeta):
    """
    Create Y variables from meta data
    :param Ymeta:
    :return:
    """
    return np.array([y.get('type') for y in Ymeta])


def create_Z(Ymeta, verbose=False):
    Ytype = extract_type(Ymeta)
    classes = list(set(Ytype))

    for y in Ytype:
        assert isinstance(y, str) and len(y) > 0, f'Label value not acceptable: "{y}"'

    if verbose:
        print('--- Classes ---')
        print('\n'.join(classes))
    Yfilter = np.array(transform_aanvraag_labels(Ytype))
    classes = list(set(Yfilter))
    if verbose:
        print('--- reducing ---')
        print('\n'.join(classes))
    assert Ytype.shape == Yfilter.shape

    # Yfilter = Yfilter.reshape(-1, 1)
    # assert Yfilter.ndim == 2
    return Yfilter
