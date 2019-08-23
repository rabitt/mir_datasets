from __future__ import absolute_import

import os
import sys

from mirdata import download_utils

import pytest

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
    from pathlib import Path
else:
    builtin_module_name = '__builtin__'
    from pathlib2 import Path


@pytest.fixture
def mock_file(mocker):
    return mocker.patch.object(download_utils, 'download_from_remote')


@pytest.fixture
def mock_untar(mocker):
    return mocker.patch.object(download_utils, 'untar')


@pytest.fixture
def mock_unzip(mocker):
    return mocker.patch.object(download_utils, 'unzip')


@pytest.fixture
def mock_path(mocker, mock_file):
    return mocker.patch.object(Path, 'mkdir')


def test_downloader(mocker, mock_path):
    mock_zip = mocker.patch.object(download_utils, 'download_zip_file')
    mock_tar = mocker.patch.object(download_utils, 'download_tar_file')
    mock_file = mocker.patch.object(download_utils, 'download_from_remote')
    # Zip only
    download_utils.downloader('a', zip_downloads=['foo'])
    mock_zip.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mocker.resetall()

    # Zip and using sub_dir flag
    download_utils.downloader('a', zip_downloads=['foo'], cleanup=True, use_subdir=True)
    mock_zip.assert_called_once_with('foo', 'a', False, cleanup=True, use_subdir=True)
    mocker.resetall()

    # tar only
    download_utils.downloader('a', tar_downloads=['foo'])
    mock_tar.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mocker.resetall()

    # Zip and using sub_dir flag
    download_utils.downloader('a', tar_downloads=['foo'], cleanup=True, use_subdir=True)
    mock_tar.assert_called_once_with('foo', 'a', False, cleanup=True, use_subdir=True)
    mocker.resetall()

    # file only
    download_utils.downloader('a', file_downloads=['foo'])
    mock_file.assert_called_once_with('foo', 'a', False)
    mocker.resetall()

    # zip and tar
    download_utils.downloader('a', zip_downloads=['foo'], tar_downloads=['foo'])
    mock_zip.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mock_tar.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mocker.resetall()

    # zip and file
    download_utils.downloader('a', zip_downloads=['foo'], file_downloads=['foo'])
    mock_zip.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mock_file.assert_called_once_with('foo', 'a', False)
    mocker.resetall()

    # tar and file
    download_utils.downloader('a', tar_downloads=['foo'], file_downloads=['foo'])
    mock_tar.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mock_file.assert_called_once_with('foo', 'a', False)
    mocker.resetall()

    # zip and tar and file
    download_utils.downloader(
        'a', zip_downloads=['foo'], tar_downloads=['foo'], file_downloads=['foo']
    )
    mock_zip.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mock_file.assert_called_once_with('foo', 'a', False)
    mock_tar.assert_called_once_with('foo', 'a', False, cleanup=False, use_subdir=False)
    mock_file.assert_called_once_with('foo', 'a', False)


def test_download_from_remote(httpserver, tmpdir):
    httpserver.serve_content(open('tests/resources/remote.wav').read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav',
        url=httpserver.url,
        checksum=('3f77d0d69dc41b3696f074ad6bf2852f'),
    )

    download_path = download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))
    expected_download_path = os.path.join(str(tmpdir), 'remote.wav')
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content('File not found!', 404)

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename='remote.wav', url=httpserver.url, checksum=('1234')
    )

    with pytest.raises(IOError):
        download_utils.download_from_remote(TEST_REMOTE, str(tmpdir))


def test_unzip(tmpdir):
    download_utils.unzip('tests/resources/remote.zip', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


def test_untar(tmpdir):
    download_utils.untar('tests/resources/remote.tar.gz', str(tmpdir))

    expected_file_location = os.path.join(str(tmpdir), 'remote.wav')
    assert os.path.exists(expected_file_location)


def test_download_zip_file(mocker, mock_file, mock_unzip):
    mock_file.return_value = "foo"
    download_utils.download_zip_file("a", "b", True)

    mock_file.assert_called_once_with("a", "b", True)
    mock_unzip.assert_called_once_with("foo", "b", cleanup=False)


def test_download_tar_file(mocker, mock_file, mock_untar):
    mock_file.return_value = "foo"
    download_utils.download_tar_file("a", "b", True)

    mock_file.assert_called_once_with("a", "b", True)
    mock_untar.assert_called_once_with("foo", "b", cleanup=False)
