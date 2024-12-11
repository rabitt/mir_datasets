# mirdata
Common loaders for Music Information Retrieval (MIR) datasets. Find the API documentation [here](https://mirdata.readthedocs.io/).

![CI status](https://github.com/mir-dataset-loaders/mirdata/actions/workflows/ci.yml/badge.svg?branch=master)
![Formatting status](https://github.com/mir-dataset-loaders/mirdata/actions/workflows/formatting.yml/badge.svg?branch=master)
![Linting status](https://github.com/mir-dataset-loaders/mirdata/actions/workflows/lint-python.yml/badge.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/mirdata/badge/?version=latest)](https://mirdata.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/mir-dataset-loaders/mirdata.svg)


[![PyPI version](https://badge.fury.io/py/mirdata.svg)](https://badge.fury.io/py/mirdata)
[![codecov](https://codecov.io/gh/mir-dataset-loaders/mirdata/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-dataset-loaders/mirdata)
[![Downloads](https://static.pepy.tech/badge/mirdata)](https://pepy.tech/project/mirdata)
[![DOI](https://zenodo.org/badge/DOI/zenodo.10070589.svg)](https://doi.org/10.5281/zenodo.10070589)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


This library provides tools for working with common MIR datasets, including tools for:
* downloading datasets to a common location and format
* validating that the files for a dataset are all present 
* loading annotation files to a common format, consistent with the format required by [mir_eval](https://github.com/craffel/mir_eval)
* parsing track level metadata for detailed evaluations


### Installation

To install, simply run:

```python
pip install mirdata
```

### Quick example
```python
import mirdata

orchset = mirdata.initialize('orchset')
orchset.download()  # download the dataset
orchset.validate()  # validate that all the expected files are there

example_track = orchset.choice_track()  # choose a random example track
print(example_track)  # see the available data
```

Or using the CLI:
```bash
python -m mirdata orchset  # download and validate the dataset
```

See the [documentation](https://mirdata.readthedocs.io/) for more examples and the API reference.


### Currently supported datasets


Supported datasets include [AcousticBrainz](https://zenodo.org/record/2553414#.X8jTgulKhhE), [DALI](https://github.com/gabolsgabs/DALI), [Guitarset](http://github.com/marl/guitarset/), [MAESTRO](https://magenta.tensorflow.org/datasets/maestro), [TinySOL](https://www.orch-idea.org/), among many others.

For the **complete list** of supported datasets, see the [documentation](https://mirdata.readthedocs.io/en/stable/source/quick_reference.html)


### Citing


There are two ways of citing mirdata:

If you are using the library for your work, please cite the version you used as indexed at Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10070589.svg)](https://doi.org/10.5281/zenodo.10070589)

If you refer to mirdata's design principles, motivation etc., please cite the following [paper](https://zenodo.org/record/3527750#.X-Inp5NKhUI):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3527750.svg)](https://doi.org/10.5281/zenodo.3527750)

```
"mirdata: Software for Reproducible Usage of Datasets"
Rachel M. Bittner, Magdalena Fuentes, David Rubinstein, Andreas Jansson, Keunwoo Choi, and Thor Kell
in International Society for Music Information Retrieval (ISMIR) Conference, 2019
```

```
@inproceedings{
  bittner_fuentes_2019,
  title={mirdata: Software for Reproducible Usage of Datasets},
  author={Bittner, Rachel M and Fuentes, Magdalena and Rubinstein, David and Jansson, Andreas and Choi, Keunwoo and Kell, Thor},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2019}
}
```

When working with datasets, please cite the version of `mirdata` that you are using (given by the `DOI` above) **AND** include the reference of the dataset, which can be found in the respective dataset loader using the `cite()` method. 

### Contributing a new dataset loader

We welcome contributions to this library, especially new datasets. Please see [contributing](https://mirdata.readthedocs.io/en/latest/source/contributing.html) for guidelines.
