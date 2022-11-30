# Timing analysis

A repository for standardizing timing analysis and data combination work with a Jupyter notebook framework and corresponding tools. 

`timing_analysis` includes tools and notebook templates to facilitate transparency and reproducibility in timing pulsars using PINT. Configuration (`.yaml`) files contain relatively compact metadata to capture decisions made during the timing process so that such information can naturally be version controlled. Configuration files can be "plugged into" standardized notebook templates to asses and update results.

More information about available tools and use cases coming soon!

Getting started
---------------

PINT is necessary for core functionality of timing_analysis, but the following packages are also required to do detailed outlier inspection and run noise analyses:

- enterprise
- enterprise_extensions
- enterprise_outliers
- pypulse

There are instructions for installing these packages and setting up your environment here: https://github.com/ipta/pulsar-env. Note that installing `mamba` must be done from a clean (base) conda environment; you can make a fresh environment with, e.g.: `conda create --name myenv`.

Installation
------------

`timing_analysis` is now available on PyPI, so users who do not wish to develop code can grab the latest tagged version with:

```
$ pip install timing_analysis
```

You may also access the latest development (not tagged) version of the repository by cloning it from GitHub, then installing:

```
$ git clone https://github.com/nanograv/timing_analysis.git
$ cd timing_analysis
$ pip install .
```

To further develop `timing_analysis` code, fork this repository, clone your fork, then:

```
$ cd timing_analysis
$ pip install -e .
$ git remote add upstream https://github.com/nanograv/timing_analysis
```

Before making changes, we highly recommend using `pulsar-env` (see above) to set up a consistent environment. Submit changes for review by opening a PR from your fork.
