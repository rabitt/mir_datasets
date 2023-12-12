import pytest

from mirdata import core, initialize, list_datasets


def test_list_datasets():
    dataset_list = list_datasets()
    assert isinstance(dataset_list, list)
    assert "beatles" in dataset_list
    assert "orchset" in dataset_list
    assert "saraga_carnatic" in dataset_list


def test_initialize():
    d = initialize("orchset")
    assert isinstance(d, core.Dataset)
    assert d.name == "orchset"

    with pytest.raises(ValueError):
        initialize("asdfasdfasdfa")
