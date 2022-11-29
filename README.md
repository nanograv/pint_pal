# Timing analysis

A long-lived repository for NANOGrav Timing analysis work.

`timing_analysis` includes tools and notebook templates to facilitate transparency and reproducibility in timing pulsars using PINT. Configuration (`.yaml`) files contain relatively compact metadata to capture decisions made during the timing process so that such information can naturally be version controlled. Config files can be plugged into standardized notebook templates to asses and update results. Many of the underlying tools are agnostic to TOA type (e.g. narrowband/wideband).

For more information about available tools and use cases, go here [under construction].

Getting started
---------------

PINT is necessary for core functionality of timing_analysis, but the following packages are also required to do detailed outlier inspection and run noise analyses:

- enterprise
- enterprise_extensions
- enterprise_outliers
- pypulse

There are instructions for installing these packages and set up your environment here: https://github.com/ipta/pulsar-env.

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

There are more instructions for developing code here [under construction], including further suggestions for setting up a standard environment, preferred workflow for opening pull requests, etc.
