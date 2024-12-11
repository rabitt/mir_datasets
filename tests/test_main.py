from unittest import mock

import pytest

from mirdata.__main__ import main, build_cli


def get_args(args=()):
    parser = build_cli()
    return parser.parse_args(args=args)

@pytest.fixture
def downloader():
    with mock.patch('mirdata.download_utils.downloader') as mock_downloader:
        yield mock_downloader

@pytest.fixture
def dataset():
    with mock.patch('mirdata.__main__.initialize') as mock_dataset:
        yield mock_dataset.return_value


def test_flags_none():
    args = get_args()

    assert args.dataset == []
    assert args.output == None
    assert args.list == False
    assert args.validate == True
    assert args.force == None
    assert args.version == 'default'


def test_flags_license():
    args = get_args(['--license'])

    assert args.license == True
    assert args.citation == False
    assert args.download == None
    assert args.validate == True

    args = get_args(['-L'])

    assert args.license == True
    assert args.citation == False


def test_flags_citation():
    args = get_args(['--citation'])

    assert args.citation == True
    assert args.license == False
    assert args.download == None

    args = get_args(['-c'])

    assert args.citation == True
    assert args.license == False
    assert args.download == None


def test_default_invocation(downloader):
    with mock.patch('mirdata.__main__._list_datasets_to_console') as list_func:
        main([])

    downloader.assert_not_called()
    list_func.assert_called_once_with(None)


def test_cli_citation(dataset):
    """ As if invoked as `python -m mirdata maestro --citation` """
    main(['maestro'], citation=True)

    dataset.cite.assert_called_once()
    dataset.license.assert_not_called()
    dataset.download.assert_not_called()
    dataset.validate.assert_not_called()


def test_cli_license(dataset):
    """ As if invoked as `python -m mirdata maestro --citation` """
    main(['maestro'], license=True)

    dataset.license.assert_called_once()
    dataset.cite.assert_not_called()
    dataset.download.assert_not_called()
    dataset.validate.assert_not_called()


def test_cli_one_dataset(dataset):
    main(['maestro'])
    dataset.download.assert_called_once()
    dataset.validate.assert_called_once()

def test_cli_multiple_datasets(dataset):
    main(['maestro', 'orchset'])

    assert dataset.download.call_count == 2
    assert dataset.validate.call_count == 2

def test_cli_all_flags(dataset):
    main(['maestro'], download=True, license=True, citation=True)

    dataset.download.assert_called_once()
    dataset.validate.assert_called_once()
    dataset.license.assert_called_once()
    dataset.cite.assert_called_once()
