#!/usr/bin/env python

"""
Command-line interface for mirdata

Examples:

    list available datasets
    $ python -m mirdata --list

    download one dataset, but skip validation
    $ python -m mirdata 'orchset' --no-validate

    download one or more datasets
    $ python -m mirdata orchset gtzan_genre
"""
import argparse
import logging
from pathlib import Path

from mirdata.core import Dataset
from . import list_datasets, initialize

logger = logging.getLogger('mirdata')


def build_cli(parser=None):
    """ Builds the command-line interface by append arguments to an argparser. """
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('dataset', nargs='*',
                        help="name of the dataset to download/validate")
    parser.add_argument('--output', '-o', default=None, type=Path,
                        help=f"target directory where datasets will be downloaded to (default: {Dataset.DEFAULT_DIR()})")
    parser.add_argument('--list', '-l', action='store_true',
                        help="list all available datasets")
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                        help='skip dataset validation')
    parser.add_argument('--force', '-f',
                        help="overwrite dataset if it exists")
    parser.add_argument('--version', '-v', default='default',
                        help="dataset version")
    parser.add_argument('--citation', '-c', action='store_true',
                        help="Only print the citation, don't download. Can be combined with --license and --download")
    parser.add_argument('--license', '-L', action='store_true',
                        help="Only print the license, don't download. Can be combined with --citation and --download")
    parser.add_argument("--download", "-d", action='store_true', default=None)

    return parser


def main(dataset, list=False, data_home=None, force=False, version='default', **kwargs):
    if list or len(dataset) == 0:
        _list_datasets_to_console(data_home)
        return

    if len(dataset) > 1:
        print(f"Preparing download of {dataset}")

    succeeded, failed = [], []
    for d in dataset:
        try:
            _download_one(d, force=force, data_home=data_home, version=version, **kwargs)
        except Exception:
            logger.error("Failed to download dataset: %s", d, exc_info=True)
            failed.append(d)
        else:
            succeeded.append(d)

    if failed:
        print("Failed to download datasets:")
        print(", ".join(failed))

    return len(failed)


def _download_one(dataset, force=False, data_home=None, version=None,
                  download=None, validate=None, citation=False, license=False,
                  **kwargs):
    print(f"Preparing {dataset} version={version or 'default'}")
    dataset = initialize(dataset, data_home=data_home, version=version)

    if download is None and (citation or license):
        download = False
    else:
        download = True

    if validate is None and download:
        validate = True

    if download:
        dataset.download(force_overwrite=force)
    if validate:
        dataset.validate()
    if license:
        dataset.license()
    if citation:
        dataset.cite()



def _list_datasets_to_console(downloaded:Path=None):
    if downloaded is not None and downloaded.exists():
        print("Downloaded datasets")
        print("-------------------")
        subdirectories = filter(Path.is_dir, downloaded.iterdir())
        print("\n".join(map(str, subdirectories)))
        print("\n\nAvailable datasets")
        print("--------------------")

    print("\n".join(list_datasets()))


if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_cli()
    args = parser.parse_args()

    return_code = main(**vars(args))
    sys.exit(return_code)
