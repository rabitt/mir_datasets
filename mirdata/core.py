"""Core mirdata classes."""
import json
import os
import random
import types
from typing import Any, List, Optional

import numpy as np
from smart_open import open

from mirdata import download_utils, validate

MAX_STR_LEN = 100
DOCS_URL = "https://mirdata.readthedocs.io/en/stable/source/mirdata.html"
DISCLAIMER = """
******************************************************************************************
DISCLAIMER: mirdata is a software package with its own license which is independent from
this dataset's license. We don not take responsibility for possible inaccuracies in the
license information provided in mirdata. It is the user's responsibility to be informed
and respect the dataset's license.
******************************************************************************************
"""

##### decorators ######


class cached_property(object):
    """Cached propery decorator.

    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj: Any, cls: type) -> Any:
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def docstring_inherit(parent):
    """Decorator function to inherit docstrings from the parent class.

    Adds documented Attributes from the parent to the child docs.
    """

    def inherit(obj):
        spaces = "    "
        if not str(obj.__doc__).__contains__("Attributes:"):
            obj.__doc__ += "\n" + spaces + "Attributes:\n"
        obj.__doc__ = str(obj.__doc__).rstrip() + "\n"
        for attribute in parent.__doc__.split("Attributes:\n")[-1].lstrip().split("\n"):
            obj.__doc__ += spaces * 2 + str(attribute).lstrip().rstrip() + "\n"

        return obj

    return inherit


##### Core Classes #####


class Dataset(object):
    """Mirdata Dataset class.

    Attributes:
        data_home (str): path where mirdata will look for the dataset
        version (str):
        name (str): the identifier of the dataset
        bibtex (str or None): dataset citation/s in bibtex format
        indexes (dict or None):
        remotes (dict or None): data to be downloaded
        readme (str): information about the dataset
        track (function): a function mapping a track_id to a mirdata.core.Track
        multitrack (function): a function mapping a mtrack_id to a mirdata.core.Multitrack
    """

    def __init__(
        self,
        data_home=None,
        version="default",
        name=None,
        track_class=None,
        multitrack_class=None,
        bibtex=None,
        indexes=None,
        remotes=None,
        download_info=None,
        license_info=None,
    ):
        """Dataset init method.

        Args:
            data_home (str or None): path where mirdata will look for the dataset
            name (str or None): the identifier of the dataset
            track_class (mirdata.core.Track or None): a Track class
            multitrack_class (mirdata.core.Multitrack or None): a Multitrack class
            bibtex (str or None): dataset citation/s in bibtex format
            remotes (dict or None): data to be downloaded
            download_info (str or None): download instructions or caveats
            license_info (str or None): license of the dataset
        """
        self.name = name
        self.data_home = self.default_path if data_home is None else data_home

        if version not in indexes:
            raise ValueError(
                "Invalid version {}. Must be one of {}.".format(version, indexes.keys())
            )

        if isinstance(indexes[version], str):
            self.version = indexes[version]
        else:
            self.version = version

        self._index_data = indexes[self.version]
        self.index_path = self._index_data.get_path(self.data_home)

        self._track_class = track_class
        self._multitrack_class = multitrack_class
        self.bibtex = bibtex
        self.remotes = remotes
        self._download_info = download_info
        self._license_info = license_info
        self.readme = "{}#module-mirdata.datasets.{}".format(DOCS_URL, self.name)

        # this is a hack to be able to have dataset-specific docstrings
        self.track = lambda track_id: self._track(track_id)
        self.track.__doc__ = self._track_class.__doc__  # set the docstring
        self.multitrack = lambda mtrack_id: self._multitrack(mtrack_id)
        self.multitrack.__doc__ = self._multitrack_class.__doc__  # set the docstring

    def __repr__(self):
        repr_string = "The {} dataset\n".format(self.name)
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        repr_string += "Call the .cite method for bibtex citations.\n"
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        if self._track_class is not None:
            repr_string += self.track.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"
        if self._multitrack_class is not None:
            repr_string += self.multitrack.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"

        return repr_string

    @cached_property
    def _index(self):
        try:
            with open(self.index_path, encoding="utf-8") as fhandle:
                index = json.load(fhandle)
        except FileNotFoundError:
            if self._index_data.remote:
                raise FileNotFoundError(
                    "This dataset's index must be downloaded. Did you run .download()?"
                )
            raise FileNotFoundError(
                f"Dataset index for {self.name} was expected "
                + "to be packaged with mirdata, but not found."
            )

        return index

    @cached_property
    def _metadata(self):
        return None

    @property
    def default_path(self):
        """Get the default path for the dataset.

        Returns:
            str: Local path to the dataset
        """
        mir_datasets_dir = os.path.join(os.getenv("HOME", "/tmp"), "mir_datasets")
        return os.path.join(mir_datasets_dir, self.name)

    def _track(self, track_id):
        """Load a track by track_id.

        Hidden helper function that gets called as a lambda.

        Args:
            track_id (str): track id of the track

        Returns:
           Track: a Track object
        """
        if self._track_class is None:
            raise AttributeError("This dataset does not have tracks")
        else:
            return self._track_class(
                track_id, self.data_home, self.name, self._index, lambda: self._metadata
            )

    def _multitrack(self, mtrack_id):
        """Load a multitrack by mtrack_id.

        Hidden helper function that gets called as a lambda.

        Args:
            mtrack_id (str): mtrack id of the multitrack

        Returns:
            MultiTrack: an instance of this dataset's MultiTrack object
        """
        if self._multitrack_class is None:
            raise AttributeError("This dataset does not have multitracks")
        else:
            return self._multitrack_class(
                mtrack_id,
                self.data_home,
                self.name,
                self._index,
                self._track_class,
                lambda: self._metadata,
            )

    def load_tracks(self):
        """Load all tracks in the dataset.

        Returns:
            dict:
                {`track_id`: track data}

        Raises:
            NotImplementedError: If the dataset does not support Tracks
        """
        return {track_id: self.track(track_id) for track_id in self.track_ids}

    def load_multitracks(self):
        """Load all multitracks in the dataset.

        Returns:
            dict:
                {`mtrack_id`: multitrack data}

        Raises:
            NotImplementedError: If the dataset does not support Multitracks
        """
        return {mtrack_id: self.multitrack(mtrack_id) for mtrack_id in self.mtrack_ids}

    def choice_track(self):
        """Choose a random track.

        Returns:
            Track: a Track object instantiated by a random track_id
        """
        return self.track(random.choice(self.track_ids))

    def choice_multitrack(self):
        """Choose a random multitrack.

        Returns:
            Multitrack: a Multitrack object instantiated by a random mtrack_id
        """
        return self.multitrack(random.choice(self.mtrack_ids))

    def _get_partitions(self, items, splits, seed, partition_names=None):
        """Helper function to get the indexes needed to split a set of ids into partitions
        Args:
            items (list): list of items to partition
            splits (list of float): a list of floats that should sum up 1. It will return as many splits as elements in the list
            seed (int): the seed used for the random generator, in order to enhance reproducibility.
            partition_names (list): list of keys to use in the output dictionary
        Returns:
            dict: a dictionary containing the partitions
        """
        if not np.isclose(np.sum(splits), 1):
            raise ValueError(
                "Splits values should sum up to 1. Given {} sums {}".format(splits, np.sum(splits))
            )

        if partition_names and len(partition_names) != len(splits):
            raise ValueError(
                "If partition_names is provided, it should have the same length as splits"
            )

        rng = np.random.default_rng(seed=seed)
        shuffled_items = rng.permutation(items)

        if not partition_names:
            partition_names = np.arange(len(splits))

        # Method from https://stackoverflow.com/a/14281094
        cdf = np.cumsum(splits)
        partitions = list(map(lambda x: int(np.ceil(x)), cdf * len(items)))
        return {
            name: shuffled_items[a:b]
            for name, a, b in zip(partition_names, [0] + partitions, partitions)
        }

    def get_track_splits(self):
        """Get predetermined track splits (e.g. train/ test) released alongside
        this dataset.

        Raises:
            AttributeError: If this dataset does not have tracks
            NotImplementedError: If this dataset does not have predetermined splits

        Returns:
            dict: splits, keyed by split name and with values of lists of track_ids
        """
        if self._track_class is None:
            raise AttributeError("This dataset does not have tracks")

        if not hasattr(self.choice_track(), "split"):
            raise NotImplementedError(
                f"The {self.name} dataset does not have an official split. Use"
                " get_random_track_splits instead."
            )

        splits = {}
        for track_id in self.track_ids:
            track = self.track(track_id)
            if track.split in splits:
                splits[track.split].append(track_id)
            else:
                splits[track.split] = [track_id]
        return splits

    def get_random_track_splits(self, splits, seed=42, split_names=None):
        """Split the tracks into partitions e.g. training, validation, test.

        Args:
            splits (list of float): a list of floats that should sum up 1. It will return as many splits as elements in the list
            seed (int): the seed used for the random generator, in order to enhance reproducibility. Defaults to 42
            split_names (list): list of keys to use in the output dictionary

        Returns:
            dict: a dictionary containing the elements in each split
        """
        if self._track_class is None:
            raise AttributeError("This dataset does not have tracks")

        return self._get_partitions(self.track_ids, splits, seed, split_names)

    def get_mtrack_splits(self):
        """Get predetermined multitrack splits (e.g. train/ test) released
        alongside this dataset.

        Raises:
            AttributeError: If this dataset does not have multitracks
            NotImplementedError: If this dataset does not have predetermined splits

        Returns:
            dict: splits, keyed by split name and with values of lists of mtrack_ids
        """
        if self._multitrack_class is None:
            raise AttributeError("This dataset does not have multitracks")

        if not hasattr(self.choice_multitrack(), "split"):
            raise NotImplementedError(
                f"The {self.name} dataset does not have an official split. Use"
                " get_random_mtrack_splits instead."
            )

        splits = {}
        for mtrack_id in self.mtrack_ids:
            mtrack = self.multitrack(mtrack_id)
            if mtrack.split in splits:
                splits[mtrack.split].append(mtrack_id)
            else:
                splits[mtrack.split] = [mtrack_id]

        return splits

    def get_random_mtrack_splits(self, splits, seed=42, split_names=None):
        """Split the multitracks into partitions, e.g. training, validation,
        test.

        Args:
            splits (list of float): a list of floats that should sum up 1. It will return as many splits as elements in the list
            seed (int): the seed used for the random generator, in order to enhance reproducibility. Defaults to 42
            split_names (list): list of keys to use in the output dictionary

        Returns:
            dict: a dictionary containing the elements in each split
        """

        if self._multitrack_class is None:
            raise AttributeError("This dataset does not have multitracks")

        return self._get_partitions(self.mtrack_ids, splits, seed)

    def cite(self):
        """Print the reference."""
        print("========== BibTeX ==========")
        print(self.bibtex)

    def license(self):
        """Print the license."""
        print("========== License ==========")
        print(self._license_info)
        print(DISCLAIMER)

    def download(
        self,
        partial_download=None,
        force_overwrite=False,
        cleanup=False,
        allow_invalid_checksum=False,
    ):
        """Download data to `save_dir` and optionally print a message.

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.
            allow_invalid_checksum (bool):
                Allow invalid checksums of the downloaded data. Useful sometimes behind some
                proxies that inspection the downloaded data. When having a different checksum
                promts a warn instead of raising an exception

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected
        """
        download_utils.downloader(
            self.data_home,
            remotes=self.remotes,
            index=self._index_data,
            partial_download=partial_download,
            info_message=self._download_info,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
            allow_invalid_checksum=allow_invalid_checksum,
        )

    @cached_property
    def track_ids(self):
        """Return track ids.

        Returns:
            list: A list of track ids
        """
        if "tracks" not in self._index:
            raise AttributeError("This dataset does not have tracks")
        return list(self._index["tracks"].keys())

    @cached_property
    def mtrack_ids(self):
        """Return track ids.

        Returns:
            list: A list of track ids
        """
        if "multitracks" not in self._index:
            raise AttributeError("This dataset does not have multitracks")
        return list(self._index["multitracks"].keys())

    def validate(self, verbose=True):
        """Validate if the stored dataset is a valid version.

        Args:
            verbose (bool): If False, don't print output

        Returns:
            * list - files in the index but are missing locally
            * list - files which have an invalid checksum
        """
        missing_files, invalid_checksums = validate.validator(
            self._index, self.data_home, verbose=verbose
        )
        return missing_files, invalid_checksums


class Track(object):
    """Track base class.

    See the docs for each dataset loader's Track class for details
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        """Track init method. Sets boilerplate attributes, including:

        - ``track_id``
        - ``_dataset_name``
        - ``_data_home``
        - ``_track_paths``
        - ``_track_metadata``

        Args:
            track_id (str): track id
            data_home (str): path where mirdata will look for the dataset
            dataset_name (str): the identifier of the dataset
            index (dict): the dataset's file index
            metadata (function or None): a function returning a dictionary of metadata or None
        """
        if track_id not in index["tracks"]:
            raise ValueError("{} is not a valid track_id in {}".format(track_id, dataset_name))
        self._metadata = metadata
        self.track_id = track_id
        self._dataset_name = dataset_name

        self._data_home = data_home
        self._track_paths = index["tracks"][track_id]

    @cached_property
    def _track_metadata(self):
        metadata = self._metadata()
        if metadata and self.track_id in metadata:
            return metadata[self.track_id]
        elif metadata:
            return metadata
        raise AttributeError("This Track does not have metadata.")

    def __repr__(self):
        properties = [v for v in dir(self.__class__) if not v.startswith("_")]
        attributes = [v for v in dir(self) if not v.startswith("_") and v not in properties]

        repr_str = "Track(\n"

        for attr in attributes:
            val = getattr(self, attr)
            if isinstance(val, str):
                if len(val) > MAX_STR_LEN:
                    val = "...{}".format(val[-MAX_STR_LEN:])
                val = '"{}"'.format(val)
            repr_str += "  {}={},\n".format(attr, val)

        for prop in properties:
            val = getattr(self.__class__, prop)
            if isinstance(val, types.FunctionType):
                continue

            if val.__doc__ is None:
                doc = ""
            else:
                doc = val.__doc__

            val_type_str = doc.split(":")[0]
            repr_str += "  {}: {},\n".format(prop, val_type_str)

        repr_str += ")"
        return repr_str

    def to_jams(self):
        raise NotImplementedError

    def get_path(self, key):
        """Get absolute path to track audio and annotations. Returns None if
        the path in the index is None.

        Args:
            key (string): Index key of the audio or annotation type

        Returns:
            str or None: joined path string or None
        """
        if self._track_paths[key][0] is None:
            return None
        else:
            return os.path.join(self._data_home, self._track_paths[key][0])


class MultiTrack(Track):
    """MultiTrack class.

    A multitrack class is a collection of track objects and their
    associated audio that can be mixed together. A multitrack is itself
    a Track, and can have its own associated audio (such as a mastered
    mix), its own metadata and its own annotations.
    """

    def __init__(self, mtrack_id, data_home, dataset_name, index, track_class, metadata):
        """Multitrack init method. Sets boilerplate attributes, including:

        - ``mtrack_id``
        - ``_dataset_name``
        - ``_data_home``
        - ``_multitrack_paths``
        - ``_multitrack_metadata``

        Args:
            mtrack_id (str): multitrack id
            data_home (str): path where mirdata will look for the dataset
            dataset_name (str): the identifier of the dataset
            index (dict): the dataset's file index
            metadata (function or None): a function returning a dictionary of metadata or None
        """
        if mtrack_id not in index["multitracks"]:
            raise ValueError("{} is not a valid mtrack_id in {}".format(mtrack_id, dataset_name))

        self.mtrack_id = mtrack_id
        self._dataset_name = dataset_name

        self._data_home = data_home
        self._multitrack_paths = index["multitracks"][self.mtrack_id]
        self._metadata = metadata
        self._track_class = track_class

        self._index = index
        self.track_ids = self._index["multitracks"][self.mtrack_id]["tracks"]

    @property
    def tracks(self):
        return {
            t: self._track_class(
                t, self._data_home, self._dataset_name, self._index, self._metadata
            )
            for t in self.track_ids
        }

    @property
    def track_audio_property(self):
        raise NotImplementedError("Mixing is not supported for this dataset")

    @cached_property
    def _multitrack_metadata(self):
        metadata = self._metadata()
        if metadata and self.mtrack_id in metadata:
            return metadata[self.mtrack_id]
        elif metadata:
            return metadata
        raise AttributeError("This MultiTrack does not have metadata")

    def get_path(self, key):
        """Get absolute path to multitrack audio and annotations. Returns None
        if the path in the index is None.

        Args:
            key (string): Index key of the audio or annotation type

        Returns:
            str or None: joined path string or None
        """
        if self._multitrack_paths[key][0] is None:
            return None
        else:
            return os.path.join(self._data_home, self._multitrack_paths[key][0])

    def get_target(self, track_keys, weights=None, average=True, enforce_length=True):
        """Get target which is a linear mixture of tracks.

        Args:
            track_keys (list): list of track keys to mix together
            weights (list or None): list of positive scalars to be used in the average
            average (bool): if True, computes a weighted average of the tracks
                if False, computes a weighted sum of the tracks
            enforce_length (bool): If True, raises ValueError if the tracks are
                not the same length. If False, pads audio with zeros to match the length
                of the longest track

        Returns:
            np.ndarray: target audio with shape (n_channels, n_samples)

        Raises:
            ValueError:
                if sample rates of the tracks are not equal
                if enforce_length=True and lengths are not equal
        """
        signals = []
        lengths = []
        sample_rates = []
        for k in track_keys:
            audio, sample_rate = getattr(self.tracks[k], self.track_audio_property)
            # ensure all signals are shape (n_channels, n_samples)
            if len(audio.shape) == 1:
                audio = audio[np.newaxis, :]
            signals.append(audio)
            lengths.append(audio.shape[1])
            sample_rates.append(sample_rate)

        if len(set(sample_rates)) > 1:
            raise ValueError(
                "Sample rates for tracks {} are not equal: {}".format(track_keys, sample_rates)
            )

        max_length = np.max(lengths)
        if any([l != max_length for l in lengths]):
            if enforce_length:
                raise ValueError(
                    "Track's {} audio are not the same length {}. Use enforce_length=False to pad"
                    " with zeros.".format(track_keys, lengths)
                )
            else:
                # pad signals to the max length
                signals = [
                    np.pad(signal, ((0, 0), (0, max_length - signal.shape[1])))
                    for signal in signals
                ]

        if weights is None:
            weights = np.ones((len(track_keys),))

        target = np.average(signals, axis=0, weights=weights)
        if not average:
            target *= np.sum(weights)

        return target

    def get_random_target(self, n_tracks=None, min_weight=0.3, max_weight=1.0):
        """Get a random target by combining a random selection of tracks with
        random weights.

        Args:
            n_tracks (int or None): number of tracks to randomly mix. If None, uses all tracks
            min_weight (float): minimum possible weight when mixing
            max_weight (float): maximum possible weight when mixing

        Returns:
            * np.ndarray - mixture audio with shape (n_samples, n_channels)
            * list - list of keys of included tracks
            * list - list of weights used to mix tracks
        """
        tracks = list(self.tracks.keys())
        assert len(tracks) > 0
        if n_tracks is not None and n_tracks < len(tracks):
            tracks = np.random.choice(tracks, n_tracks, replace=False)

        weights = np.random.uniform(low=min_weight, high=max_weight, size=len(tracks))
        target = self.get_target(tracks, weights=weights)
        return target, tracks, weights

    def get_mix(self):
        """Create a linear mixture given a subset of tracks.

        Args:
            track_keys (list): list of track keys to mix together

        Returns:
            np.ndarray: mixture audio with shape (n_samples, n_channels)
        """
        tracks = list(self.tracks.keys())
        assert len(tracks) > 0
        return self.get_target(tracks)


class Index(object):
    """Class for storing information about dataset indexes.

    Args:
        filename (str): The index filename (not path), e.g. "example_dataset_index_1.2.json"
        url (str or None): None if index is not remote, or a url to download from
        checksum (str or None): None if index is not remote, or the md5 checksum of the file
        partial_download (list or None): if provided, specifies a subset of Dataset.remotes
            corresponding to this index to be downloaded. If None, all Dataset.remotes will
            be downloaded when calling Dataset.download()

    Attributes:
        remote (download_utils.RemoteFileMetadata or None): None if index is not remote, or
            a RemoteFileMetadata object
        partial_download (list or None): a list of keys to partially download, or None
    """

    def __init__(
        self,
        filename: str,
        url: Optional[str] = None,
        checksum: Optional[str] = None,
        partial_download: Optional[List[str]] = None,
    ):
        self.filename = filename
        self.remote: Optional[download_utils.RemoteFileMetadata]
        if url and checksum:
            self.remote = download_utils.RemoteFileMetadata(
                filename=filename,
                url=url,
                checksum=checksum,
                destination_dir="mirdata_indexes",
            )
        elif url or checksum:
            raise ValueError("Remote indexes must have both a url and a checksum specified.")
        else:
            self.remote = None

        self.partial_download = partial_download

    def get_path(self, data_home: str) -> str:
        """Get the absolute path to the index file.

        Args:
            data_home (str): Path where the dataset's data lives

        Returns:
            str: absolute path to the index file
        """
        # if the index is downloaded from remote, it is in the same folder
        # as the data
        if self.remote:
            return os.path.join(data_home, "mirdata_indexes", self.filename)
        # if the index is part of mirdata locally, it is in the indexes folder
        # of the repository
        else:
            return os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "datasets/indexes",
                self.filename,
            )
