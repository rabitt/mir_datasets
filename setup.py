""" Setup script for mir_datasets. """
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

version_sfl = SourceFileLoader("mirdata.version", "mirdata/version.py")
version_module = version_sfl.load_module()

with open("README.md", "r") as fdesc:
    long_description = fdesc.read()

if __name__ == "__main__":
    setup(
        name="mirdata",
        version=version_module.version,
        description="Common loaders for MIR datasets.",
        url="https://github.com/mir-dataset-loaders/mirdata",
        packages=find_packages(exclude=["test", "*.test", "*.test.*"]),
        download_url="http://github.com/mir-dataset-loaders/mirdata/releases",
        package_data={"mirdata": ["datasets/indexes/*"]},
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Multimedia :: Sound/Audio :: Analysis",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        keywords="mir dataset loader audio",
        license="BSD-3-Clause",
        install_requires=[
            "black == 22.8.0",  # last version for python3.6
            "tqdm",
            "librosa >= 0.9.2",
            "numpy>=1.16",
            "jams",
            "requests",
            "pretty_midi >= 0.2.8",
            "chardet",
            "pyyaml",
            "scipy",
            "h5py >= 2.10.0",
            "smart_open >= 5.0.0",
            "Deprecated >= 1.2.13",
            "pandas >= 1.0.0",
        ],
        extras_require={
            "tests": [
                "pytest>=4.4.0",
                "pytest-cov>=2.6.1",
                "pytest-pep8>=1.0.0",
                "pytest-mock>=1.10.1",
                "pytest-localserver>=0.5.0",
                "testcontainers>=2.3",
                "future==0.17.1",
                "coveralls>=1.7.0",
                "types-PyYAML",
                "types-chardet",
                "smart_open[all] >= 5.0.0",
                "openpyxl==3.0.7",
                "dali-dataset==1.1",
                "music21==6.7.1",
            ],
            "docs": [
                "numpydoc",
                "recommonmark",
                "sphinx>=3.4.0",
                "sphinxcontrib-napoleon",
                "sphinx_rtd_theme",
            ],
            "compmusic_hindustani_rhythm": ["openpyxl==3.0.7"],
            "dali": ["dali-dataset==1.1"],
            "compmusic_carnatic_rhythm": ["openpyxl==3.0.7"],
            "haydn_op20": ["music21==6.7.1"],
            "gcs": ["smart_open[gcs]"],
            "s3": ["smart_open[s3]"],
            "http": ["smart_open[http]"],
        },
    )
