"""Setup script for mir_datasets."""
from importlib.machinery import SourceFileLoader

from setuptools import find_packages, setup

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
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        keywords="mir dataset loader audio",
        license="BSD-3-Clause",
        install_requires=[
            "attrs>=23.1.0",
            "black>=23.3.0",
            "chardet>=5.0.0",
            "Deprecated>=1.2.14",
            "h5py>=3.7.0",
            "jams>=0.3.4",
            "librosa>=0.10.1",
            "numpy>=1.21.6",
            "pandas>=1.3.5",
            "pretty_midi>=0.2.10",
            "pyyaml>=6.0",
            "requests>=2.31.0",
            "scipy>=1.7.3",
            "tqdm>=4.66.1",
        ],
        extras_require={
            "tests": [
                "decorator>=5.1.1",
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
            ],
            "docs": [
                "numpydoc",
                "recommonmark",
                "sphinx>=3.4.0",
                "sphinxcontrib-napoleon",
                "sphinx_rtd_theme",
                "sphinx-togglebutton",
            ],
            "compmusic_hindustani_rhythm": ["openpyxl==3.0.10"],
            "dali": ["dali-dataset==1.1"],
            "compmusic_carnatic_rhythm": ["openpyxl==3.0.10"],
            "haydn_op20": ["music21==6.7.1"],
            "cipi": ["music21==6.7.1"],
            "gcs": ["smart_open[gcs]"],
            "s3": ["smart_open[s3]"],
            "http": ["smart_open[http]"],
        },
    )
