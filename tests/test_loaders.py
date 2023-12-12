import importlib
import inspect
import io
import os
import sys

import pytest
import requests

import mirdata
from mirdata import core
from tests.test_utils import get_attributes_and_properties

DATASETS = mirdata.DATASETS
CUSTOM_TEST_TRACKS = {
    "beatles": "0111",
    "cante100": "008",
    "compmusic_carnatic_rhythm": "10003",
    "compmusic_carnatic_varnam": "dharini_abhogi",
    "compmusic_hindustani_rhythm": "20001",
    "compmusic_indian_tonic": "0a6ebaa4-87cc-452d-a7af-a2006e96f16a_0-180",
    "compmusic_jingju_acappella": "lseh-Tan_Yang_jia-Hong_yang_dong-qm",
    "compmusic_otmm_makam": "cafcdeaf-e966-4ff0-84fb-f660d2b68365",
    "compmusic_raga": "Aruna_Sairam.Valli_Kanavan",
    "four_way_tabla": "AHK_solo-tintal-1",
    "giantsteps_key": "3",
    "dali": "4b196e6c99574dd49ad00d56e132712b",
    "da_tacos": "coveranalysis#W_163992#P_547131",
    "filosax": "multitrack_02_sax_1",
    "freesound_one_shot_percussive_sounds": "183",
    "giantsteps_tempo": "113",
    "gtzan_genre": "country.00000",
    "guitarset": "03_BN3-119-G_solo",
    "idmt_smt_audio_effects": "G73-45200-3341-33944",
    "irmas": "1",
    "medley_solos_db": "d07b1fc0-567d-52c2-fef4-239f31c9d40e",
    "medleydb_melody": "MusicDelta_Beethoven",
    "mridangam_stroke": "224030",
    "rwc_classical": "RM-C003",
    "rwc_jazz": "RM-J004",
    "rwc_popular": "RM-P001",
    "openmic2018": "000046_3840",
    "salami": "2",
    "saraga_carnatic": "116_Bhuvini_Dasudane",
    "saraga_hindustani": "50_Irani_Bhairavi_Thumri",
    "tinysol": "Fl-ord-C4-mf-N-T14d",
    "dagstuhl_choirset": "DCS_LI_QuartetB_Take04_B2",
    "tonas": "01-D_AMairena",
}

TEST_DATA_HOME = os.path.normpath("tests/resources/mir_datasets")


def test_dataset_attributes():
    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        assert (
            dataset.name == dataset_name
        ), "{}.dataset attribute does not match dataset name".format(dataset_name)
        assert dataset.bibtex is not None, "No BIBTEX information provided for {}".format(
            dataset_name
        )
        assert dataset._license_info is not None, "No LICENSE information provided for {}".format(
            dataset_name
        )
        assert (
            isinstance(dataset.remotes, dict) or dataset.remotes is None
        ), "{}.REMOTES must be a dictionary".format(dataset_name)
        assert isinstance(dataset._index, dict), "{}.DATA is not properly set".format(dataset_name)
        assert (
            isinstance(dataset._download_info, str) or dataset._download_info is None
        ), "{}.DOWNLOAD_INFO must be a string".format(dataset_name)
        assert type(dataset._track_class) == type(
            core.Track
        ), "{}.Track must be an instance of core.Track".format(dataset_name)
        assert callable(dataset.download), "{}.download is not a function".format(dataset_name)


def test_smart_open():
    for dataset_name in DATASETS:
        # import module
        dataset_module = importlib.import_module(f"mirdata.datasets.{dataset_name}")
        code_lines = inspect.getsource(dataset_module)

        # check if os.path.exists is called:
        assert "os.path.exists" not in code_lines, (
            f"{dataset_name} uses os.path.exists, "
            + "which does not work with remote filesystems. Instead use a try/except. "
            + "See orchset.py for an example."
        )

        # if open is called, make sure smart open is imported
        not_overwritten_message = (
            f"{dataset_name} uses open, but does not overwrite it with smart_open. "
            + "Add the line `from smart_open import open` to the top of the module."
        )
        overwritten_wrong_message = (
            f"{dataset_name} overwrites open using something other than smart_open. "
            + "Add the line `from smart_open import open` to the top of the module."
        )
        if "open(" in code_lines:
            assert hasattr(dataset_module, "open"), not_overwritten_message
            assert (
                dataset_module.open.__module__ == "smart_open.smart_open_lib"
            ), overwritten_wrong_message


def test_cite_and_license():
    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.cite()
        sys.stdout = sys.__stdout__

        text_trap = io.StringIO()
        sys.stdout = text_trap
        dataset.license()
        sys.stdout = sys.__stdout__


KNOWN_ISSUES = {}  # key is module, value is REMOTE key
DOWNLOAD_EXCEPTIONS = ["maestro", "slakh", "gtzan_genre"]


def test_download(mocker):
    for dataset_name in DATASETS:
        print(dataset_name)
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # test parameters & defaults
        assert callable(dataset.download), "{}.download is not callable".format(dataset_name)
        params = inspect.signature(dataset.download).parameters

        expected_params = [
            ("partial_download", None),
            ("force_overwrite", False),
            ("cleanup", False),
        ]
        for exp in expected_params:
            assert exp[0] in params, "{}.download must have {} as a parameter".format(
                dataset_name, exp[0]
            )
            assert (
                params[exp[0]].default == exp[1]
            ), "The default value of {} in {}.download must be {}".format(
                dataset_name, exp[0], exp[1]
            )

        # check that the download method can be called without errors
        if dataset.remotes != {}:
            mocker.patch.object(dataset, "remotes")
            if dataset_name not in DOWNLOAD_EXCEPTIONS:
                try:
                    dataset.download()
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

                mocker.resetall()

            # check that links are online
            for key in dataset.remotes:
                # skip this test if it's in known issues
                if dataset_name in KNOWN_ISSUES and key in KNOWN_ISSUES[dataset_name]:
                    continue

                url = dataset.remotes[key].url
                try:
                    request = requests.head(url)
                    assert request.ok, "Link {} for {} does not return OK".format(url, dataset_name)
                except requests.exceptions.ConnectionError:
                    assert False, "Link {} for {} is unreachable".format(url, dataset_name)
                except:
                    assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
        else:
            try:
                dataset.download()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


# This is magically skipped by the the remote fixture `skip_local` in conftest.py
# when tests are run with the --local flag
def test_validate(skip_local):
    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        try:
            dataset.validate()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        try:
            dataset.validate(verbose=False)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])


def test_load_and_trackids():
    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        try:
            track_ids = dataset.track_ids
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
        assert type(track_ids) is list, "{}.track_ids() should return a list".format(dataset_name)
        trackid_len = len(track_ids)
        # if the dataset has tracks, test the loaders
        if dataset._track_class is not None:
            try:
                choice_track = dataset.choice_track()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])
            assert isinstance(
                choice_track, core.Track
            ), "{}.choice_track must return an instance of type core.Track".format(dataset_name)

            try:
                dataset_data = dataset.load_tracks()
            except:
                assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

            assert isinstance(dataset_data, dict), "{}.load should return a dictionary".format(
                dataset_name
            )
            assert len(dataset_data.keys()) == trackid_len, (
                "the dictionary returned {}.load() does not have the same number of elements as"
                " {}.track_ids()".format(dataset_name, dataset_name)
            )


def test_track():
    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # if the dataset doesn't have a track object, make sure it raises a value error
        # and move on to the next dataset
        if dataset._track_class is None:
            with pytest.raises(NotImplementedError):
                dataset.track("~faketrackid~?!")
            continue

        if dataset_name in CUSTOM_TEST_TRACKS:
            trackid = CUSTOM_TEST_TRACKS[dataset_name]
        else:
            trackid = dataset.track_ids[0]

        # test data home specified
        try:
            track_test = dataset.track(trackid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            track_test, core.Track
        ), "{}.track must be an instance of type core.Track".format(dataset_name)

        assert hasattr(track_test, "to_jams"), "{}.track must have a to_jams method".format(
            dataset_name
        )

        # test calling all attributes, properties and cached properties
        track_data = get_attributes_and_properties(track_test)

        for attr in track_data["attributes"]:
            getattr(track_test, attr)

        for prop in track_data["properties"]:
            getattr(track_test, prop)

        for cprop in track_data["cached_properties"]:
            getattr(track_test, cprop)

        # Validate JSON schema
        try:
            jam = track_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.track({})".format(
            dataset_name, trackid
        )

        # will fail if something goes wrong with __repr__
        try:
            text_trap = io.StringIO()
            sys.stdout = text_trap
            print(track_test)
            sys.stdout = sys.__stdout__
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        with pytest.raises(ValueError):
            dataset.track("~faketrackid~?!")


# This tests the case where there is no data in data_home.
# It makes sure that the track can be initialized and the
# attributes accessed, but that anything requiring data
# files errors (all properties and cached properties).
def test_track_placeholder_case():
    data_home_dir = "not/a/real/path"

    for dataset_name in DATASETS:
        print(dataset_name)
        data_home = os.path.join(data_home_dir, dataset_name)
        dataset = mirdata.initialize(dataset_name, data_home, version="test")

        if not dataset._track_class:
            continue

        if dataset_name in CUSTOM_TEST_TRACKS:
            trackid = CUSTOM_TEST_TRACKS[dataset_name]
        else:
            trackid = dataset.track_ids[0]

        try:
            track_test = dataset.track(trackid)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        track_data = get_attributes_and_properties(track_test)

        for attr in track_data["attributes"]:
            getattr(track_test, attr)

        for prop in track_data["properties"]:
            with pytest.raises(Exception):
                getattr(track_test, prop)

        for cprop in track_data["cached_properties"]:
            with pytest.raises(Exception):
                getattr(track_test, cprop)


# for load_* functions which require more than one argument
# module_name : {function_name: {parameter2: value, parameter3: value}}
EXCEPTIONS = {
    "dali": {"load_annotations_granularity": {"granularity": "notes"}},
    "guitarset": {
        "load_pitch_contour": {"string_num": 1},
        "load_notes": {"string_num": 1},
        "load_chords": {"leadsheet_version": False},
    },
    "tonas": {"load_f0": {"corrected": True}},
    "compmusic_carnatic_varnam": {
        "load_notation": {
            "taala_path": "a/fake/path",
            "structure_path": "a/fake/path",
        },
    },
}
SKIP = {
    "acousticbrainz_genre": [
        "load_all_train",
        "load_all_validation",
        "load_tagtraum_validation",
        "load_tagtraum_train",
        "load_allmusic_train",
        "load_allmusic_validation",
        "load_lastfm_train",
        "load_lastfm_validation",
        "load_discogs_train",
        "load_discogs_validation",
    ],
    "baf": ["load_matches"],
}


def test_load_methods():
    for dataset_name in DATASETS:
        dataset_module = importlib.import_module(f"mirdata.datasets.{dataset_name}")

        all_methods = dir(dataset_module)
        load_methods = [getattr(dataset_module, m) for m in all_methods if m.startswith("load_")]
        for load_method in load_methods:
            method_name = load_method.__name__

            # skip overrides, add to the SKIP dictionary to skip a specific load method
            if dataset_name in SKIP and method_name in SKIP[dataset_name]:
                continue

            if load_method.__doc__ is None:
                raise ValueError(
                    "mirdata.datasets.{}.Dataset.{} has no documentation".format(
                        dataset_name, method_name
                    )
                )

            # add to the EXCEPTIONS dictionary above if your load_* function needs
            # more than one argument.
            if dataset_name in EXCEPTIONS and method_name in EXCEPTIONS[dataset_name]:
                extra_params = EXCEPTIONS[dataset_name][method_name]
                with pytest.raises(IOError):
                    load_method("a/fake/filepath", **extra_params)
            else:
                with pytest.raises(IOError):
                    load_method("a/fake/filepath")


CUSTOM_TEST_MTRACKS = {}


def test_multitracks():
    data_home_dir = os.path.normpath("tests/resources/mir_datasets")

    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # TODO this is currently an opt-in test. Make it an opt out test
        # once #265 is addressed
        if dataset_name in CUSTOM_TEST_MTRACKS:
            mtrack_id = CUSTOM_TEST_MTRACKS[dataset_name]
        else:
            # there are no multitracks
            continue

        try:
            dataset.MultiTrack(mtrack_id)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        # test data home specified
        data_home = os.path.join(data_home_dir, dataset_name)
        dataset_specific = mirdata.initialize(dataset_name, data_home=data_home)
        try:
            mtrack_test = dataset_specific.MultiTrack(mtrack_id, data_home=data_home)
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert isinstance(
            mtrack_test, core.MultiTrack
        ), "{}.MultiTrack must be an instance of type core.MultiTrack".format(dataset_name)

        assert hasattr(mtrack_test, "to_jams"), "{}.MultiTrack must have a to_jams method".format(
            dataset_name
        )

        # Validate JSON schema
        try:
            jam = mtrack_test.to_jams()
        except:
            assert False, "{}: {}".format(dataset_name, sys.exc_info()[0])

        assert jam.validate(), "Jams validation failed for {}.MultiTrack({})".format(
            dataset_name, mtrack_id
        )


def test_random_splits():
    split = [0.9, 0.1]
    for dataset_name in DATASETS:
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

        # check wrong type of split function
        if dataset._track_class is None:
            with pytest.raises(AttributeError):
                dataset.get_random_track_splits(split)

        if dataset._multitrack_class is None:
            with pytest.raises(AttributeError):
                dataset.get_random_mtrack_splits(split)

        # check splits for tracks
        if dataset._track_class:
            splits = dataset.get_random_track_splits(split)
            assert len(dataset.track_ids) == sum([len(i) for i in splits.values()])

        # check splits for multitracks
        if dataset._multitrack_class:
            splits = dataset.get_random_mtrack_splits(split)
            assert len(dataset.mtrack_ids) == sum([len(i) for i in splits.values()])


def test_predetermined_splits():
    required_track = ["irmas", "mtg_jamendo_autotagging_moodtheme", "slakh", "tinysol"]
    required_mtrack = ["slakh"]
    for dataset_name in DATASETS:
        print(dataset_name)
        dataset = mirdata.initialize(
            dataset_name, os.path.join(TEST_DATA_HOME, dataset_name), version="test"
        )

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
        except (AttributeError, NotImplementedError):
            assert dataset_name not in required_track

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
        except (AttributeError, NotImplementedError):
            assert dataset_name not in required_mtrack
