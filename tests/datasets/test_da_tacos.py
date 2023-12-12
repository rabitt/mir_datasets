# -*- coding: utf-8 -*-
import os

import numpy as np
import pytest

from mirdata.datasets import da_tacos
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "coveranalysis#W_163992#P_547131"
    data_home = os.path.normpath("tests/resources/mir_datasets/da_tacos")
    dataset = da_tacos.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "cens_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5",
        ),
        "crema_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5",
        ),
        "hpcp_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_hpcp/W_163992_hpcp/P_547131_hpcp.h5",
        ),
        "key_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_key/W_163992_key/P_547131_key.h5",
        ),
        "madmom_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_madmom/W_163992_madmom/P_547131_madmom.h5",
        ),
        "mfcc_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_mfcc/W_163992_mfcc/P_547131_mfcc.h5",
        ),
        "tags_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/da_tacos/"),
            "da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5",
        ),
        "track_id": "coveranalysis#W_163992#P_547131",
        "performance_id": "P_547131",
        "subset": "coveranalysis",
        "label": "W_163992",
        "work_title": "Trade Winds, Trade Winds",
        "work_artist": "Aki Aleong",
        "performance_title": "Trade Winds, Trade Winds",
        "performance_artist": "Aki Aleong",
        "release_year": "1961",
        "work_id": "W_163992",
        "is_instrumental": False,
        "performance_artist_mbid": "9bfa011f-8331-4c9a-b49b-d05bc7916605",
        "mb_performances": {
            "4ce274b3-0979-4b39-b8a3-5ae1de388c4a": {"length": "175000"},
            "7c10ba3b-6f1d-41ab-8b20-14b2567d384a": {"length": "177653"},
        },
    }

    expected_property_types = {
        "cens": np.ndarray,
        "crema": np.ndarray,
        "hpcp": np.ndarray,
        "key": dict,
        "madmom": dict,
        "mfcc": np.ndarray,
        "tags": list,
    }
    run_track_tests(track, expected_attributes, expected_property_types)


def test_to_jams():
    default_trackid = "coveranalysis#W_163992#P_547131"
    data_home = "tests/resources/mir_datasets/da_tacos"
    dataset = da_tacos.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    jam = track.to_jams()
    assert len(jam["sandbox"].keys()) == 11
    assert "work_id" in jam["sandbox"]
    assert "performance_id" in jam["sandbox"]
    assert "label" in jam["sandbox"]
    assert "subset" in jam["sandbox"]
    assert "cens" in jam["sandbox"]
    assert "crema" in jam["sandbox"]
    assert "hpcp" in jam["sandbox"]
    assert "key" in jam["sandbox"]
    assert "madmom" in jam["sandbox"]
    assert "mfcc" in jam["sandbox"]
    assert "tags" in jam["sandbox"]


def test_load_tags():
    tags_path = "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5"
    tags_data = da_tacos.load_tags(tags_path)

    assert isinstance(tags_data, list)
    assert len(tags_data) == 50
    assert da_tacos.load_tags(None) is None


def test_load_cens():
    cens_path = "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_cens/W_163992_cens/P_547131_cens.h5"
    cens_data = da_tacos.load_cens(cens_path)

    assert isinstance(cens_data, np.ndarray)

    assert cens_data.shape[0] == 15227
    assert cens_data.shape[1] == 12


def test_load_crema():
    crema_path = "tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_crema/W_163992_crema/P_547131_crema.h5"
    crema_data = da_tacos.load_crema(crema_path)

    assert isinstance(crema_data, np.ndarray)

    assert crema_data.shape[0] == 12
    assert crema_data.shape[1] == 15226


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/da_tacos"
    dataset = da_tacos.Dataset(data_home)
    metadata = dataset._metadata
    default_trackid = "coveranalysis#W_163992#P_547131"
    assert metadata[default_trackid] == {
        "work_title": "Trade Winds, Trade Winds",
        "work_artist": "Aki Aleong",
        "perf_title": "Trade Winds, Trade Winds",
        "perf_artist": "Aki Aleong",
        "release_year": "1961",
        "work_id": "W_163992",
        "perf_id": "P_547131",
        "instrumental": "No",
        "perf_artist_mbid": "9bfa011f-8331-4c9a-b49b-d05bc7916605",
        "mb_performances": {
            "4ce274b3-0979-4b39-b8a3-5ae1de388c4a": {"length": "175000"},
            "7c10ba3b-6f1d-41ab-8b20-14b2567d384a": {"length": "177653"},
        },
    }


def test_load_metadata_not_there():
    data_home = "asdf/asdf/mir_datasets/da_tacos"
    dataset = da_tacos.Dataset(data_home)
    with pytest.raises(FileNotFoundError):
        dataset._metadata


def test_filters():
    data_home = "tests/resources/mir_datasets/da_tacos"
    dataset = da_tacos.Dataset(data_home)

    data = dataset.filter_index("asdfasdfasdf")
    assert data == {}

    data_benchmark = dataset.benchmark_tracks()
    assert isinstance(data_benchmark, dict)
    assert data_benchmark

    data_coveranalysis = dataset.coveranalysis_tracks()
    assert isinstance(data_coveranalysis, dict)
    assert data_coveranalysis
