"""
This test takes a long time, but it makes sure that the datset can be locally downloaded,
validated successfully, and loaded.
"""

import os
import pytest
import tqdm

from tests.test_utils import get_attributes_and_properties
import mirdata


@pytest.fixture()
def dataset(test_dataset, dataset_version):
    if test_dataset == "":
        return None
    elif test_dataset not in mirdata.DATASETS:
        raise ValueError("{} is not a dataset in mirdata".format(test_dataset))
    data_home = os.path.join("tests/resources/mir_datasets_full", test_dataset)
    return mirdata.initialize(test_dataset, data_home, version=dataset_version)


# This is magically skipped by the the remote fixture `skip_remote` in conftest.py
# when tests are run without the --local flag
def test_download(skip_remote, dataset, skip_download):
    if dataset is None:
        pytest.skip()

    # download the dataset
    if not skip_download:
        dataset.download()

        print(
            "If this dataset does not have openly downloadable data, "
            + "follow the instructions printed by the download message and "
            + "rerun this test."
        )


def test_validation(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    # run validation
    missing_files, invalid_checksums = dataset.validate(verbose=True)

    assert missing_files == {
        key: {} for key in dataset._index.keys() if not key == "version"
    }
    assert invalid_checksums == {
        key: {} for key in dataset._index.keys() if not key == "version"
    }


def test_load_tracks(skip_remote, dataset):
    if dataset is None or dataset._track_class is None:
        pytest.skip()

    # run load
    all_data = dataset.load_tracks()

    assert isinstance(all_data, dict)

    track_ids = dataset.track_ids
    assert set(track_ids) == set(all_data.keys())

    # test that all attributes and properties can be called
    for track_id in tqdm.tqdm(track_ids):
        track = all_data[track_id]
        track_data = get_attributes_and_properties(track)

        for attr in track_data["attributes"]:
            ret = getattr(track, attr)

        for prop in track_data["properties"]:
            ret = getattr(track, prop)

        for cprop in track_data["cached_properties"]:
            ret = getattr(track, cprop)

        jam = track.to_jams()
        assert jam.validate()


def test_load_mtracks(skip_remote, dataset):
    if dataset is None or dataset._multitrack_class is None:
        pytest.skip()

    # run load
    all_data = dataset.load_multitracks()

    assert isinstance(all_data, dict)

    mtrack_ids = dataset.mtrack_ids
    assert set(mtrack_ids) == set(all_data.keys())

    # test that all attributes and properties can be called
    for mtrack_id in tqdm.tqdm(mtrack_ids):
        mtrack = all_data[mtrack_id]
        mtrack_data = get_attributes_and_properties(mtrack)

        for attr in mtrack_data["attributes"]:
            ret = getattr(mtrack, attr)

        for prop in mtrack_data["properties"]:
            ret = getattr(mtrack, prop)

        for cprop in mtrack_data["cached_properties"]:
            ret = getattr(mtrack, cprop)

        jam = mtrack.to_jams()
        assert jam.validate()


def test_index(skip_remote, dataset):
    if dataset is None:
        pytest.skip()

    okeys = ["tracks", "multitracks", "records"]

    if "version" not in dataset._index.keys():
        raise NotImplementedError("The top-level key 'version' is missing in the index")

    if not any(key in dataset._index.keys() for key in okeys):
        raise NotImplementedError(
            "At least one of the optional top-level keys {} should be in the index".format(
                okeys
            )
        )


def test_predetermined_splits(dataset):
    if dataset is None:
        pytest.skip()

    # test custom get_track_splits functions
    try:
        splits = dataset.get_track_splits()
        assert isinstance(splits, dict)
        used_tracks = set()
        for k in splits:
            assert all([t in dataset.track_ids for t in splits[k]])
            this_split = set(splits[k])
            assert not used_tracks.intersection(this_split)
            used_tracks.update(this_split)
        assert used_tracks == set(dataset.track_ids)
    except (AttributeError, NotImplementedError):
        pass

    # test custom get_mtrack_splits functions
    try:
        splits = dataset.get_mtrack_splits()
        assert isinstance(splits, dict)
        used_tracks = set()
        for k in splits:
            assert all([t in dataset.mtrack_ids for t in splits[k]])
            this_split = set(splits[k])
            assert not used_tracks.intersection(this_split)
            used_tracks.update(this_split)
        assert used_tracks == set(dataset.mtrack_ids)
    except (AttributeError, NotImplementedError):
        pass
