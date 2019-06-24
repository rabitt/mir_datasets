# -*- coding: utf-8 -*-
"""SALAMI Dataset Loader

SALAMI Dataset.
Details can be found at http://ddmal.music.mcgill.ca/research/salami/annotations


Attributes:
    SALAMI_DIR (str): The directory name for SALAMI dataset. Set to `'Salami'`.

    SALAMI_INDEX (dict): {track_id: track_data}.
        track_data is a `BeatlesTrack` namedtuple.

    SALAMI_METADATA (None): (todo?)

    SALAMI_ANNOT_REMOTE (RemoteFileMetadata (namedtuple)): metadata
        of SALAMI dataset. It includes the annotation file name, annotation
        file url, and checksum of the file.

    SalamiTrack (namedtuple): (todo)

"""
from collections import namedtuple
import csv
import numpy as np
import os

import mirdata.utils as utils

SALAMI_INDEX = utils.load_json_index('salami_index.json')
SALAMI_METADATA = None
SALAMI_DIR = 'Salami'
SALAMI_ANNOT_REMOTE = utils.RemoteFileMetadata(
    filename='salami-data-public-master.zip',
    url='https://github.com/DDMAL/salami-data-public/archive/master.zip',
    checksum='b01d6eb5b71cca1f3163fae4b2cd4c61',
)

SalamiTrack = namedtuple(
    'SalamiTrack',
    [
        'track_id',
        'audio_path',
        'sections_annotator_1_uppercase',
        'sections_annotator_1_lowercase',
        'sections_annotator_2_uppercase',
        'sections_annotator_2_lowercase',
        'source',
        'annotator_1_id',
        'annotator_2_id',
        'duration_sec',
        'title',
        'artist',
        'annotator_1_time',
        'annotator_2_time',
        'broad_genre',
        'genre',
    ],
)


def download(data_home=None, force_overwrite=False):
    """Download SALAMI Dataset (annotations).
    The audio files are not provided.

    Args:
        data_home (str): Local home path to store the dataset
        force_overwrite (bool): whether to overwrite the existing downloaded data

    """
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, SALAMI_DIR)

    if exists(data_home) and not force_overwrite:
        return

    if force_overwrite:
        utils.force_delete_all(SALAMI_ANNOT_REMOTE, dataset_path=None, data_home=data_home)

    download_path = utils.download_from_remote(
        SALAMI_ANNOT_REMOTE, data_home=data_home, force_overwrite=force_overwrite
    )
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    utils.unzip(download_path, dataset_path, cleanup=True)
    missing_files, invalid_checksums = validate(dataset_path, data_home)
    if missing_files or invalid_checksums:
        print(
            """
            Unfortunately the audio files of the Salami dataset are not available
            for download. If you have the Salami dataset, place the contents into a
            folder called Salami with the following structure:
                > Salami/
                    > salami-data-public-master/
                    > audio/
            and copy the Salami folder to {}
        """.format(
                save_path
            )
        )


def exists(data_home=None):
    """Return if SALAMI dataset folder exists

    Args:
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        (bool): True if SALAMI dataset folder exists

    """

    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, SALAMI_DIR)
    return os.path.exists(dataset_path)


def validate(dataset_path, data_home=None):
    """Validate if the stored dataset is a valid version

    Args:
        dataset_path (str): SALAMI dataset local path
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    missing_files, invalid_checksums = utils.validator(
        SALAMI_INDEX, data_home, dataset_path
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """

    return list(SALAMI_INDEX.keys())


def load(data_home=None):
    """Load SALAMI dataset

    Args:
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        (dict): {`track_id`: track data}

    """
    save_path = utils.get_save_path(data_home)
    dataset_path = os.path.join(save_path, SALAMI_DIR)

    validate(dataset_path, data_home)
    salami_data = {}
    for key in track_ids():
        salami_data[key] = load_track(key, data_home=data_home)
    return salami_data


def load_track(track_id, data_home=None):
    """Load a track data

    Args:
        track_id (str):
        data_home (str): Local home path that the dataset is being stored.

    Returns:
        SalamiTrack (todo)

    """
    if track_id not in SALAMI_INDEX.keys():
        raise ValueError('{} is not a valid track ID in Salami'.format(track_id))
    track_data = SALAMI_INDEX[track_id]

    if SALAMI_METADATA is None or SALAMI_METADATA['data_home'] != data_home:
        _reload_metadata(data_home)
        if SALAMI_METADATA is None:
            raise EnvironmentError('Could not find Salami metadata file')

    if track_id in SALAMI_METADATA.keys():
        track_metadata = SALAMI_METADATA[track_id]
    else:
        # annotations with missing metadata
        track_metadata = {
            'source': None,
            'annotator_1_id': None,
            'annotator_2_id': None,
            'duration_sec': None,
            'title': None,
            'artist': None,
            'annotator_1_time': None,
            'annotator_2_time': None,
            'class': None,
            'genre': None,
        }
    salami_path = utils.get_local_path(data_home, SALAMI_DIR)
    annotations_dir = os.path.join(
        salami_path, 'salami-data-public-master', 'annotations'
    )
    annotators = [
        any(SALAMI_INDEX[track_id]['annotator_1_uppercase']),
        any(SALAMI_INDEX[track_id]['annotator_2_uppercase']),
    ]
    all_annotators_section_data = _load_sections(
        utils.get_local_path(annotations_dir, track_id), annotators
    )

    return SalamiTrack(
        track_id,
        utils.get_local_path(data_home, track_data['audio'][0]),
        all_annotators_section_data[0],
        all_annotators_section_data[1],
        all_annotators_section_data[2],
        all_annotators_section_data[3],
        track_metadata['source'],
        track_metadata['annotator_1_id'],
        track_metadata['annotator_2_id'],
        track_metadata['duration_sec'],
        track_metadata['title'],
        track_metadata['artist'],
        track_metadata['annotator_1_time'],
        track_metadata['annotator_2_time'],
        track_metadata['class'],
        track_metadata['genre'],
    )


def _load_sections(sections_path, annotators):
    all_annotators_section_data = []
    for a in range(len(annotators)):
        for f in ['uppercase.txt', 'lowercase.txt']:
            times, secs = [], []
            if annotators[a]:
                file_path = os.path.join(
                    sections_path, 'parsed', 'textfile{}_{}'.format(str(a + 1), f)
                )
                if os.path.exists(file_path):

                    with open(file_path, 'r') as fhandle:
                        reader = csv.reader(fhandle, delimiter='\t')
                        for line in reader:
                            times.append(float(line[0]))
                            secs.append(line[1])
                    times, secs = np.array(times), np.array(secs)
                    # remove sections with length == 0
                    times_revised = np.delete(times, np.where(np.diff(times) == 0))
                    secs_revised = np.delete(secs, np.where(np.diff(times) == 0))
                    all_annotators_section_data.append(
                        utils.SectionData(
                            np.array(times_revised[:-1]),
                            np.array(times_revised)[1:],
                            np.array(secs_revised)[:-1],
                        )
                    )
                else:
                    all_annotators_section_data.append(None)
            else:
                all_annotators_section_data.append(None)

    return all_annotators_section_data


def _load_metadata(data_home):

    metadata_path = utils.get_local_path(
        data_home,
        os.path.join(
            SALAMI_DIR, 'salami-data-public-master', 'metadata', 'metadata.csv'
        ),
    )

    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        raw_data = []
        for line in reader:
            if line[0] == 'SONG ID':
                continue
            raw_data.append(line)

    metadata_index = {}
    for line in raw_data:
        track_id = line[0]

        metadata_index[track_id] = {
            'source': line[1],
            'annotator_1_id': line[2],
            'annotator_2_id': line[3],
            'duration_sec': line[5],
            'title': line[7],
            'artist': line[8],
            'annotator_1_time': line[10],
            'annotator_2_time': line[11],
            'class': line[14],
            'genre': line[15],
        }

    metadata_index['data_home'] = data_home

    return metadata_index


def _reload_metadata(data_home):
    global SALAMI_METADATA
    SALAMI_METADATA = _load_metadata(data_home=data_home)


def cite():
    """Print the reference"""

    cite_data = """
===========  MLA ===========
Smith, Jordan Bennett Louis, et al.,
"Design and creation of a large-scale database of structural annotations",
12th International Society for Music Information Retrieval Conference (2011)

========== Bibtex ==========
@inproceedings{smith2011salami,
    title={Design and creation of a large-scale database of structural annotations.},
    author={Smith, Jordan Bennett Louis and Burgoyne, John Ashley and
          Fujinaga, Ichiro and De Roure, David and Downie, J Stephen},
    booktitle={12th International Society for Music Information Retrieval Conference},
    year={2011},
    series = {ISMIR},
}
"""

    print(cite_data)
