import numpy as np


# Create Y variables from meta data
def extract_type(Ymeta):
    return np.array([y.get('type') for y in Ymeta])


# All possible classes to "aanvraag" and "other"
def filter_Y(Ytype):
    return ['aanvraag' if 'aanvraag' in x else 'other' for x in Ytype]


def create_Y(Ymeta, verbose=False):
    Ytype = extract_type(Ymeta)
    classes = list(set(Ytype))
    if verbose:
        print('--- Classes ---')
        print('\n'.join(classes))
    Yfilter = np.array(filter_Y(Ytype))
    classes = list(set(Yfilter))
    if verbose:
        print('--- reducing ---')
        print('\n'.join(classes))
    assert Ytype.shape == Yfilter.shape
    return Yfilter
