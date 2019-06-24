# -*- coding: utf-8 -*-
"""MedleyDB pitch Dataset Loader

MedleyDB is a dataset of annotated, royalty-free multitrack recordings.
MedleyDB was curated primarily to support research on melody extraction,
addressing important shortcomings of existing collections. For each song
we provide melody f0 annotations as well as instrument activations for
evaluating automatic instrument recognition.

Details can be found at https://medleydb.weebly.com


Attributes:
    MEDLEYDB_PITCH_INDEX (dict): {track_id: track_data}.
        track_data is a `MedleydbPitchTrack` namedtuple.

    MEDLEYDB_PITCH_DIR (str): The directory name for MedleyDB melody dataset.
        Set to `'MedleyDB-Pitch'`.

    MEDLEYDB_METADATA (None): TODO

    MedleydbPitchTrack (namedtuple): namedtuple to store the metadata of a MedleyDB track (melody).
        Tuple names: `'track_id', 'melody1', 'melody2', 'melody3', 'audio_path',
            'artist', 'title', 'genre', 'is_excerpt', 'is_instrumental', 'n_sources'`.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import csv
import json
import numpy as np
import os

import mirdata.utils as utils

MEDLEYDB_PITCH_INDEX = utils.load_json_index('medleydb_pitch_index.json')
MEDLEYDB_PITCH_DIR = 'MedleyDB-Pitch'
MEDLEYDB_METADATA = None

MedleydbPitchTrack = namedtuple(
    'MedleydbPitchTrack',
    ['track_id', 'pitch', 'audio_path', 'instrument', 'artist', 'title', 'genre'],
)


def download(data_home=None):
    """MedleyDB is not available for downloading directly.
    This function prints a helper message to download MedleyDB
    through zenodo.org.

    Args:
        data_home (str): Local home path to store the dataset

    """

    save_path = utils.get_save_path(data_home)

    print(
        """
        To download this dataset, visit:
        https://zenodo.org/record/2620624#.XKZc7hNKh24
        and request access.

        Once downloaded, unzip the file MedleyDB-Pitch.zip
        and place the result in:
        {save_path}
    """.format(
            save_path=save_path
        )
    )


def validate(dataset_path, data_home=None):
    """Validate if the stored dataset is a valid version

    Args:
        dataset_path (str): MedleyDB pitch dataset local path
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """

    missing_files, invalid_checksums = utils.validator(
        MEDLEYDB_PITCH_INDEX, data_home, dataset_path
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """

    return list(MEDLEYDB_PITCH_INDEX.keys())


def load(data_home=None):
    """Load MedleyDB pitch dataset

    Args:
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        (dict): {`track_id`: track data}

    """

    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, MEDLEYDB_PITCH_DIR)

    validate(dataset_path, data_home)
    medleydb_pitch_data = {}
    for key in track_ids():
        medleydb_pitch_data[key] = load_track(key, data_home=data_home)
    return medleydb_pitch_data


def load_track(track_id, data_home=None):
    """Load a track data

    Args:
        track_id (str): track id to load
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        MedleydbPitchTrack (todo)

    """

    if track_id not in MEDLEYDB_PITCH_INDEX.keys():
        raise ValueError(
            '{} is not a valid track ID in MedleyDB-Pitch'.format(track_id)
        )
    track_data = MEDLEYDB_PITCH_INDEX[track_id]

    if MEDLEYDB_METADATA is None or MEDLEYDB_METADATA['data_home'] != data_home:
        _reload_metadata(data_home)
        if MEDLEYDB_METADATA is None:
            raise EnvironmentError('Could not find MedleyDB-Pitch metadata file')

    track_metadata = MEDLEYDB_METADATA[track_id]

    pitch_data = _load_pitch(utils.get_local_path(data_home, track_data['pitch'][0]))

    return MedleydbPitchTrack(
        track_id,
        pitch_data,
        utils.get_local_path(data_home, track_data['audio'][0]),
        track_metadata['instrument'],
        track_metadata['artist'],
        track_metadata['title'],
        track_metadata['genre'],
    )


def _load_pitch(pitch_path):
    if not os.path.exists(pitch_path):
        return None
    times = []
    freqs = []
    confidence = []
    with open(pitch_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            freqs.append(float(line[1]))
            confidence.append(0 if line[1] == '0' else 1)

    melody_data = utils.F0Data(np.array(times), np.array(freqs), np.array(confidence))
    return melody_data


def _reload_metadata(data_home):
    global MEDLEYDB_METADATA
    MEDLEYDB_METADATA = _load_metadata(data_home=data_home)


def _load_metadata(data_home):
    metadata_path = utils.get_local_path(
        data_home, os.path.join(MEDLEYDB_PITCH_DIR, 'medleydb_pitch_metadata.json')
    )
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    metadata['data_home'] = data_home
    return metadata


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Bittner, Rachel, et al.
"MedleyDB: A multitrack dataset for annotation-intensive MIR research."
In Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR). 2014.

========== Bibtex ==========
@inproceedings{bittner2014medleydb,
    Author = {Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan P},
    Booktitle = {International Society of Music Information Retrieval (ISMIR)},
    Month = {October},
    Title = {Medley{DB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
    Year = {2014}
}
"""

    print(cite_data)
