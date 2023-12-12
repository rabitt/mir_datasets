# This script modifies indexes from mirdata <= 0.3.0b0 to support versions, and to check integrity of
# tracks, multitracks, metadata and tables. The structure of indexes will now be:
#
# index = {
#    'version': ...,
#    'tracks': {
#        ...
#    },
#    'records': {
#        ...
#    },
#    'multitracks': {
#        ...
#    },
#    'metadata': {
#        ...
#    }
# }


import json
import os

from tqdm import tqdm

import mirdata
from mirdata.validate import md5

INDEXES_PATH = "../../mirdata/datasets/indexes/"
ALL_INDEXES = os.listdir(INDEXES_PATH)
DATASETS = [
    "beatles",
    "beatport_key",
    "cante100",
    "dali",
    "giantsteps_key",
    "giantsteps_tempo",
    "groove_midi",
    "gtzan_genre",
    "guitarset",
    "irmas",
    "maestro",
    "medley_solos_db",
    "medleydb_melody",
    "mridangam_stroke",
    "orchset",
    "rwc_classical",
    "rwc_jazz",
    "rwc_popular",
    "salami",
    "tinysol",
]


def get_metadata_paths(module):
    """Get the metadata path of each dataset.

    Parameters
    ----------
    module (str): dataset name

    Returns
    -------
    metadata_path (str or None : relative path to metadata
    """

    customized_paths = {
        "beatles": None,
        "beatport_key": None,
        "cante100": {"cante100Meta": "cante100Meta.xml"},
        "dali": {"dali_metadata": "dali_metadata.json"},
        "giantsteps_key": None,
        "giantsteps_tempo": None,
        "groove_midi": {"info": "info.csv"},
        "gtzan_genre": None,
        "guitarset": None,
        "irmas": None,
        "maestro": {"maestro-v2.0.0": "maestro-v2.0.0.json"},
        "medley_solos_db": {
            "Medley-solos-DB_metadata": os.path.join("annotation", "Medley-solos-DB_metadata.csv")
        },
        "medleydb_pitch": {"medleydb_pitch_metadata": "medleydb_pitch_metadata.json"},
        "mridangam_stroke": None,
        "orchset": {
            "Orchset - Predominant Melodic Instruments": "Orchset - Predominant Melodic Instruments.csv"
        },
        "rwc_classical": {"rwc-c": os.path.join("metadata-master", "rwc-c.csv")},
        "rwc_jazz": {"rwc-j": os.path.join("metadata-master", "rwc-j.csv")},
        "rwc_popular": {"rwc-p": os.path.join("metadata-master", "rwc-p.csv")},
        "salami": {
            "metadata": os.path.join(
                "salami-data-public-hierarchy-corrections", "metadata", "metadata.csv"
            )
        },
        "saraga": None,
        "tinysol": {"TinySOL_metadata": os.path.join("annotation", "TinySOL_metadata.csv")},
    }

    return customized_paths[module]


def get_dataset_version(module):
    """Get the version of each dataset.

    Parameters
    ----------
    module (str): dataset name

    Returns
    -------
    version (str): dataset version in mirdata
    """

    # All this websites linked to currently supported versions were accessed on 11/02/20
    customized_versions = {
        "beatles": "1.2",  # http://isophonics.net/content/reference-annotations-beatles
        "beatport_key": "1.0",  # https://zenodo.org/record/1101082/export/xd#.X6BFXpNKi3J
        "cante100": "1.0",  # https://zenodo.org/record/1324183
        "dali": "1.0",  # https://zenodo.org/record/2577915#.X6BGZJNKi3I
        "giantsteps_key": "+",  # https://zenodo.org/record/1095691#.X6BGrZNKi3J
        "giantsteps_tempo": "2.0",  # https://github.com/GiantSteps/giantsteps-tempo-dataset
        "groove_midi": "1.0.0",  # https://magenta.tensorflow.org/datasets/groove
        "gtzan_genre": None,  # http://marsyas.info/downloads/datasets.html
        "guitarset": "1.1.0",  # https://zenodo.org/record/3371780#.X6BIZpNKi3I
        "ikala": None,  # http://mac.citi.sinica.edu.tw/ikala/
        "irmas": "1.0",  # https://zenodo.org/record/1290750/
        "maestro": "2.0.0",  # https://magenta.tensorflow.org/datasets/maestro
        "medley_solos_db": "1.2",  # https://zenodo.org/record/1344103
        "medleydb_melody": "5.0",  # https://zenodo.org/record/2628782#.X6BKOJNKi3J
        "medleydb_pitch": "2.0",  # https://zenodo.org/record/2620624#.XKZc7hNKh24
        "mridangam_stroke": "1.5",  # https://zenodo.org/record/1265188#.X6BKhpNKi3J
        "orchset": "1.0",  # https://zenodo.org/record/1289786#.X6BK3pNKi3L
        "rwc_classical": None,  # https://staff.aist.go.jp/m.goto/RWC-MDB/
        "rwc_jazz": None,  # https://staff.aist.go.jp/m.goto/RWC-MDB/
        "rwc_popular": None,  # https://staff.aist.go.jp/m.goto/RWC-MDB/
        "salami": "2.0-corrected",  # https://github.com/DDMAL/salami-data-public/pull/15
        "saraga": "1.0",  # https://zenodo.org/record/1256127/#.X65rnZNKhUI
        "tinysol": "6.0",  # https://zenodo.org/record/1101082/export/xd#.X6BFXpNKi3J
    }
    return customized_versions[module]


def update_index(all_indexes):
    """Function to update indexes to new format.

    Parameters
    ----------
    all_indexes (list): list of all current dataset indexes
    """

    for index_name in tqdm(all_indexes):
        module = index_name.replace("_index.json", "")

        # load old index
        old_index = mirdata.initialize(module)._index

        # avoid modifying when running multiple times
        if "tracks" in old_index.keys():
            old_index = old_index["tracks"]

        data_home = mirdata.initialize(module).data_home

        # get metadata checksum
        metadata_files = get_metadata_paths(module)
        metadata_checksums = None

        if metadata_files is not None:
            metadata_checksums = {
                key: [
                    metadata_files[key],
                    md5(os.path.join(data_home, metadata_files[key])),
                ]
                for key in metadata_files.keys()
            }

        # get version of dataset
        version = get_dataset_version(module)

        # Some datasets have a single metadata file, some have multiple.
        # The computation of the checksum should be customized in the make_index
        # of each dataset. This is a patch to convert previous indexes to the new format.
        new_index = {"version": version, "tracks": old_index}

        if metadata_files is not None:
            new_index["metadata"] = metadata_checksums

        with open(os.path.join(INDEXES_PATH, index_name), "w") as fhandle:
            json.dump(new_index, fhandle, indent=2)


def test_index(dataset_names):
    """Test if updated indexes are as expected.

    Parameters
    ----------
    dataset_names (list): list of dataset names
    """

    mandatory_keys = ["version"]
    for module in dataset_names:
        index = mirdata.initialize(module)._index
        assert type(index["tracks"]) == dict
        assert set(mandatory_keys) <= set([*index.keys()])


def test_track_load(dataset_names):
    """Function to test all loaders work and indexes are fine (run locally)

    Parameters
    ----------
    dataset_names (list): list of dataset names
    """
    for module in dataset_names:
        dataset = mirdata.initialize(module)
        dataset.load_tracks()


def main():
    DATASETS = ["rwc_classical"]
    ALL_INDEXES = ["rwc_classical_index.json"]
    # Download metadata from all datasets for computing metadata checksums
    for module in DATASETS:
        if module not in ["dali", "beatles", "groove_midi"]:
            dataset = mirdata.initialize(module)
            if dataset.remotes is not None:
                dataset.download(
                    partial_download=[
                        "metadata" if "metadata" in dataset.remotes else key
                        for key in dataset.remotes
                        if key is not "audio" and "training" not in key and "testing" not in key
                    ]
                )

    # Update index to new format
    print("Updating indexes...\n")
    update_index(ALL_INDEXES)
    # Check new indexes are shaped as expected
    print("Quick check on datasets...\n")
    test_index(DATASETS)
    test_track_load(DATASETS)


if __name__ == "__main__":
    main()
