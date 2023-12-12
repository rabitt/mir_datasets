import importlib
import os
import pkgutil

from .version import version as __version__

DATASETS = [
    d.name for d in pkgutil.iter_modules([os.path.dirname(os.path.abspath(__file__)) + "/datasets"])
]


def list_datasets():
    """Get a list of all mirdata dataset names.

    Returns:
        list: list of dataset names as strings
    """
    return DATASETS


def initialize(dataset_name, data_home=None, version="default"):
    """Load a mirdata dataset by name.

    Example:
        .. code-block:: python

            orchset = mirdata.initialize('orchset')  # get the orchset dataset
            orchset.download()  # download orchset
            orchset.validate()  # validate orchset
            track = orchset.choice_track()  # load a random track
            print(track)  # see what data a track contains
            orchset.track_ids()  # load all track ids

    Args:
        dataset_name (str): the dataset's name
            see mirdata.DATASETS for a complete list of possibilities
        data_home (str or None): path where the data lives. If None
            uses the default location.
        version (str or None): which version of the dataset to load.
            If None, the default version is loaded.

    Returns:
        Dataset: a mirdata.core.Dataset object
    """
    if dataset_name not in DATASETS:
        raise ValueError("Invalid dataset {}".format(dataset_name))

    module = importlib.import_module("mirdata.datasets.{}".format(dataset_name))

    return module.Dataset(data_home=data_home, version=version)
