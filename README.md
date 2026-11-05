# PINT Pal

[![Notebook Pipeline (Ubuntu)](https://github.com/nanograv/pint_pal/actions/workflows/test_notebook.yml/badge.svg)](https://github.com/nanograv/pint_pal/actions/workflows/test_notebook.yml)
[![PyPi Repository](https://img.shields.io/pypi/v/pint-pal.svg)](https://pypi.org/project/pint-pal/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pint-pal.svg)](https://pypi.org/project/pint-pal/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8092498.svg)](https://doi.org/10.5281/zenodo.8092498)

A repository for standardizing timing analysis and data combination work with a Jupyter notebook framework and corresponding tools. 

`pint_pal` includes tools and notebook templates to facilitate transparency and reproducibility in timing pulsars using PINT. Configuration (`.yaml`) files contain relatively compact metadata to capture decisions made during the timing process so that such information can naturally be version controlled. Configuration files can be "plugged into" standardized notebook templates to asses and update results.

Dependencies
---------------

[PINT](https://github.com/nanograv/PINT) is necessary for core functionality of `pint_pal`. A number of other pulsar timing packages, mostly related to noise model inference, are also required:

- [Enterprise](https://github.com/nanograv/enterprise)
- [Enterprise Extensions](https://github.com/nanograv/enterprise_extensions)
- [Enterprise Outliers](https://github.com/nanograv/enterprise_outliers)
- [Discovery](https://github.com/nanograv/discovery)
- [PyPulse](https://github.com/mtlam/PyPulse)

For the most part, these should not need to be installed separately, but will be pulled in automatically when installing PINT Pal (but see the detailed installation instructions below for a note about SuiteSparse, which is required by Enterprise).
Installation
------------
For most pulsar timing users who just want a working environment with the latest version of PINT Pal, we recommend using the included [Pixi](https://pixi.prefix.dev/) environments. These are the same environments used for CI testing. PINT Pal is installed in editable mode, so these environments can easily be used for development work as well. To use them:
1. Clone this repository:
```
git clone https://github.com/nanograv/pint_pal
```
2. If you have not already done so, install Pixi:
```
wget -qO - https://pixi.sh/install.sh | sh
```
3. Install and activate the environment:
```
cd pint_pal
pixi shell
```
This creates a subshell, so to deactivate it, simply `exit`.

Pixi provides several other useful commands, which are described in its documentation. Of note:
- `pixi install` is used to install environments, but it is also run automatically the first time `pixi shell` is invoked.
- `pixi run` can be used to run commands within the environment without starting a full subshell.
- The dependencies installed into the environment are set by the `project.dependencies` table, and several `tool.pixi` tables, in `pyproject.toml`.
- Pixi environments are installed in the `.pixi` subdirectory of the `pint_pal` directory and can be several gigabytes in size.

In addition to the above, tagged releases of PINT Pal are available on PyPI and conda-forge. These can be installed using any of several well-known installation tools, including `pip`, `uv`, `conda`, `mamba`, and `pixi`. Depending on the installation method, some care needs to be taken, because PINT Pal depends on [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), which is a wrapper around the [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) library. Installation methods that use conda-forge can pull in SuiteSparse automatically, but those that use PyPI cannot, so SuiteSparse will need to be installed separately. Details on several possible installation methods are given below.
### Conda-Forge (`conda`/`mamba`/`pixi`)
When installing from conda-forge, all dependencies should be pulled in automatically.
To create a new virtual environment and install PINT Pal:

Using `conda`, run
```
conda create -n pint-pal-env pint-pal
```
Using `mamba`, run
```
mamba create -n pint-pal-env pint-pal
```
Using `pixi`, run
```
mkdir pint-pal-env && cd pint-pal-env
pixi init
pixi add pint-pal
```
### PyPI (`pip`/`uv`)
To install PINT Pal from PyPI, you will first need to install SuiteSparse using your system's package manager. The method used for this will depend on your operating system.

On Debian-based Linux distros (Debian, Ubuntu, Linux Mint), run:
```
sudo apt install libsuitesparse-dev
```
On Red Hat-based distros (Fedora, RHEL, Rocky Linux, AlmaLinux), run:
```
sudo dnf install suitesparse
```
On MacOS, use homebrew:
```
brew install suite-sparse
```

Then, create a new virtual environment and install PINT Pal.

Using `pip`:
```
python -m venv pint-pal-env
source pint-pal-env/bin/activate
pip install pint-pal
```
Using `uv`:
```
uv init pint-pal-env
cd pint-pal-env
uv add pint-pal
```

