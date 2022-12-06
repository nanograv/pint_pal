# PINT Pal

[![Notebook Pipeline (Ubuntu)](https://github.com/nanograv/pint_pal/actions/workflows/test_notebook.yml/badge.svg)](https://github.com/nanograv/pint_pal/actions/workflows/test_notebook.yml)

A repository for standardizing timing analysis and data combination work with a Jupyter notebook framework and corresponding tools. 

`pint_pal` includes tools and notebook templates to facilitate transparency and reproducibility in timing pulsars using PINT. Configuration (`.yaml`) files contain relatively compact metadata to capture decisions made during the timing process so that such information can naturally be version controlled. Configuration files can be "plugged into" standardized notebook templates to asses and update results.

More information about available tools and use cases coming soon!

Getting started
---------------

[PINT](https://github.com/nanograv/PINT) is necessary for core functionality of `pint_pal`, but the following packages are also required to do detailed outlier inspection and run noise analyses:

- [enterprise](https://github.com/nanograv/enterprise)
- [enterprise_extensions](https://github.com/nanograv/enterprise_extensions)
- [enterprise_outliers](https://github.com/nanograv/enterprise_outliers)
- [pypulse](https://github.com/mtlam/PyPulse)

There are instructions for installing these packages and setting up your environment here: https://github.com/ipta/pulsar-env. Please note that installing `mamba` must be done from a clean (base) `conda` environment; you can make and activate a fresh environment with, e.g.: 

```
$ conda create --name installer && conda activate installer
```

Installation
------------

`pint_pal` is now available on PyPI, so users who do not wish to develop code can grab the latest tagged version with:

```
$ pip install pint_pal
```

You may also access the latest development (not tagged) version of the repository by cloning it from GitHub, then installing:

```
$ git clone https://github.com/nanograv/pint_pal.git
$ cd pint_pal
$ pip install .
```

To further develop `pint_pal` code, fork this repository, clone your fork, then:

```
$ cd pint_pal
$ pip install -e .
$ git remote add upstream https://github.com/nanograv/pint_pal
```

Before making changes, we highly recommend using `pulsar-env` (see above) to set up a consistent environment. Submit changes for review by opening a PR from your fork.
