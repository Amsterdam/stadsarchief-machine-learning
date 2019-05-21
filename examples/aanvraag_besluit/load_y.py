import numpy as np


# Create Y variables from meta data
def extract_type(Ymeta):
    return np.array([y.get('type') for y in Ymeta])


# All possible classes to "aanvraag" and "other"
def filter_Y(Ytype):
    return ['aanvraag' if x == 'aanvraag' else 'other' for x in Ytype]


def create_Y(Ymeta, verbose=False):
    Ytype = extract_type(Ymeta)
    classes = list(set(Ytype))

    for y in Ytype:
        assert isinstance(y, str) and len(y) > 0, f'Label value not acceptable: "{y}"'

    if verbose:
        print('--- Classes ---')
        print('\n'.join(classes))
    Yfilter = np.array(filter_Y(Ytype))
    classes = list(set(Yfilter))
    if verbose:
        print('--- reducing ---')
        print('\n'.join(classes))
    assert Ytype.shape == Yfilter.shape

    # Yfilter = Yfilter.reshape(-1, 1)
    # assert Yfilter.ndim == 2
    return Yfilter
