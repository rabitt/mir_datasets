import os

from mirdata import utils

import json
import pytest
from unittest import mock


def test_md5():
    audio_file = b"audio1234"

    expected_checksum = "6dc00d1bac757abe4ea83308dde68aab"

    with mock.patch("builtins.open", new=mock.mock_open(read_data=audio_file)) as mock_open:
        md5_checksum = utils.md5("test_file_path")
        assert expected_checksum == md5_checksum


@pytest.mark.parametrize("test_index,expected_missing,expected_inv_checksum", [
    ("test_index_valid.json", 0, 0),
    ("test_index_missing_file.json", 1, 0),
    ("test_index_invalid_checksum.json", 0, 1),
])
def test_validator(test_index,
                   expected_missing,
                   expected_inv_checksum):
    index_path = os.path.join("tests/indexes", test_index)
    with open(index_path) as index_file:
        test_index = json.load(index_file)

    missing_files, invalid_checksums = utils.validator(test_index, "tests/resources/")

    assert expected_missing == len(missing_files)
    assert expected_inv_checksum == len(invalid_checksums)
