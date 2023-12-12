import itertools
import json
import os
import types

import pytest
from smart_open import open

import mirdata
from mirdata import validate

DEFAULT_DATA_HOME = os.path.normpath(os.path.join(os.getenv("HOME", "/tmp"), "mir_datasets"))


def run_track_tests(track, expected_attributes, expected_property_types):
    track_attr = get_attributes_and_properties(track)

    # test track attributes
    for attr in track_attr["attributes"]:
        print("{}: {}".format(attr, getattr(track, attr)))
        assert expected_attributes[attr] == getattr(track, attr)

    # test track property types
    for prop in track_attr["cached_properties"] + track_attr["properties"]:
        print("{}: {}".format(prop, type(getattr(track, prop))))
        if prop in expected_property_types:
            assert isinstance(getattr(track, prop), expected_property_types[prop])
        elif prop in expected_attributes:
            assert expected_attributes[prop] == getattr(track, prop)
        else:
            assert False, "{} not in expected_property_types or expected_attributes".format(prop)


def run_multitrack_tests(mtrack):
    tracks = getattr(mtrack, "tracks")
    track_ids = getattr(mtrack, "track_ids")
    assert list(tracks.keys()) == track_ids
    for k, track in tracks.items():
        assert getattr(track, "track_id") in track_ids


def get_attributes_and_properties(class_instance):
    attributes = []
    properties = []
    cached_properties = []
    functions = []
    for val in dir(class_instance.__class__):
        if val.startswith("_"):
            continue

        attr = getattr(class_instance.__class__, val)
        if isinstance(attr, mirdata.core.cached_property):
            cached_properties.append(val)
        elif isinstance(attr, property):
            properties.append(val)
        elif isinstance(attr, types.FunctionType):
            functions.append(val)
        else:
            raise ValueError("Unknown type {}".format(attr))

    non_attributes = list(itertools.chain.from_iterable([properties, cached_properties, functions]))
    for val in dir(class_instance):
        if val.startswith("_"):
            continue
        if val not in non_attributes:
            attributes.append(val)
    return {
        "attributes": attributes,
        "properties": properties,
        "cached_properties": cached_properties,
        "functions": functions,
    }


@pytest.fixture
def mock_validated(mocker):
    return mocker.patch.object(validate, "check_validated")


@pytest.fixture
def mock_validator(mocker):
    return mocker.patch.object(validate, "validator")


@pytest.fixture
def mock_validate_index(mocker):
    return mocker.patch.object(validate, "validate_index")


def test_md5(mocker):
    audio_file = b"audio1234"
    expected_checksum = "6dc00d1bac757abe4ea83308dde68aab"
    mocker.patch("mirdata.validate.open", new=mocker.mock_open(read_data=audio_file))
    md5_checksum = validate.md5("test_file_path")
    assert expected_checksum == md5_checksum


@pytest.mark.parametrize(
    "test_index,expected_missing,expected_inv_checksum",
    [
        ("test_index_valid.json", {"tracks": {}}, {"tracks": {}}),
        (
            "test_index_missing_file.json",
            {"tracks": {"10161_chorus": [os.path.normpath("tests/resources/10162_chorus.wav")]}},
            {"tracks": {}},
        ),
        (
            "test_index_invalid_checksum.json",
            {"tracks": {}},
            {"tracks": {"10161_chorus": [os.path.normpath("tests/resources/10161_chorus.wav")]}},
        ),
    ],
)
def test_validate_index(test_index, expected_missing, expected_inv_checksum):
    index_path = os.path.join(os.path.normpath("tests/indexes"), test_index)
    with open(index_path, "r") as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = validate.validate_index(
        test_index, os.path.normpath("tests/resources/")
    )

    assert expected_missing == missing_files
    assert expected_inv_checksum == invalid_checksums


@pytest.mark.parametrize(
    "missing_files,invalid_checksums",
    [
        (
            {"tracks": {"10161_chorus": [os.path.normpath("tests/resources/10162_chorus.wav")]}},
            {"tracks": {}},
        ),
        (
            {"tracks": {}},
            {"tracks": {"10161_chorus": [os.path.normpath("tests/resources/10161_chorus.wav")]}},
        ),
        ({"tracks": {}}, {"tracks": {}}),
    ],
)
def test_validator(mocker, mock_validate_index, missing_files, invalid_checksums):
    mock_validate_index.return_value = missing_files, invalid_checksums

    m, c = validate.validator("foo", "bar", False)
    assert m == missing_files
    assert c == invalid_checksums
    mock_validate_index.assert_called_once_with("foo", "bar", False)
