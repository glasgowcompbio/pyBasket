import gzip
import os
import pathlib
import pickle

import numpy as np
from loguru import logger

DEFAULT_NUM_CHAINS = None  # let pymc decide
DEFAULT_EFFICACY_CUTOFF = 0.90
DEFAULT_FUTILITY_CUTOFF = 0.05
DEFAULT_EARLY_FUTILITY_STOP = False
DEFAULT_EARLY_EFFICACY_STOP = False
DEFAULT_TARGET_ACCEPT = 0.99

GROUP_STATUS_OPEN = 'OPEN'
GROUP_STATUS_EARLY_STOP_FUTILE = 'EARLY_STOP_FUTILE'
GROUP_STATUS_EARLY_STOP_EFFECTIVE = 'EARLY_STOP_EFFECTIVE'
GROUP_STATUS_COMPLETED_INEFFECTIVE = 'COMPLETED_INEFFECTIVE'
GROUP_STATUS_COMPLETED_EFFECTIVE = 'COMPLETED_EFFECTIVE'

MODEL_INDEPENDENT = 'independent'
MODEL_INDEPENDENT_BERN = 'independent_bern'
MODEL_BHM = 'BHM'
MODEL_PYBASKET = 'pyBasket'

def create_if_not_exist(out_dir):
    """
    Creates a directory if it doesn't already exist
    Args:
        out_dir: the directory to create, if it doesn't exist

    Returns: None.

    """
    if not pathlib.Path(out_dir).exists():
        logger.info('Created %s' % out_dir)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


def save_obj(obj, filename):
    """
    Save object to file. This is useful for storing simulation results and other objects.

    If the directory containing the specified filename doesn't exist, it will be created first.
    The object will be saved using gzip + pickle (highest protocol).

    Args:
        obj: the Python object to save
        filename: the output filename to use

    Returns: None
    """

    # workaround for
    # TypeError: can't pickle _thread.lock objects
    # when trying to pickle a progress bar
    if hasattr(obj, 'bar'):
        obj.bar = None

    out_dir = os.path.dirname(filename)
    create_if_not_exist(out_dir)
    logger.info('Saving %s to %s' % (type(obj), filename))
    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """
    Load saved object from file

    Args:
        filename: The filename to load. Should be saved using the `save_obj` method.

    Returns: the loaded object

    """
    try:
        with gzip.GzipFile(filename, 'rb') as f:
            return pickle.load(f)
    except OSError:
        logger.warning('Old, invalid or missing pickle in %s. '
                       'Please regenerate this file.' % filename)
        raise


class Group():
    '''
    A class to represent a patient group, or basket, or arm
    '''

    def __init__(self, group_id):
        self.idx = group_id
        self.responses = []
        self.classes = []
        self.clusters = []
        self.status = GROUP_STATUS_OPEN

    def register(self, patient_data):
        self.responses.extend(patient_data.responses)
        self.classes.extend(patient_data.classes)
        self.clusters.extend(patient_data.clusters)

    @property
    def response_indices(self):
        return [self.idx] * len(self.responses)

    def __repr__(self):
        nnz = np.count_nonzero(self.responses)
        total = len(self.responses)
        return 'Group %d (%s): %d/%d' % (self.idx, self.status, nnz, total)
