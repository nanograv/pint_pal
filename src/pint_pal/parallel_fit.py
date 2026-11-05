import concurrent.futures
from astropy import units as u, constants as c
from astropy.time import Time
import numpy as np
import copy
import io
from typing import Dict, Optional
from loguru import logger as log
import pint.fitter
from pint.models import get_model_and_toas, get_model
import pint.gridutils
from pint.types import file_like
import pint.models
import pint.toa

# create a maximal model including all potential parameters
# note that this could be defined elsewhere (e.g., in a config file)
# many of them will be fixed to zero during fits
# this initial list is for all models
_extraparnames = ["PX", "F2"]
# only for ELL1 using PB
_ell1pbextraparnames = ["M2", "SINI", "PBDOT", "A1DOT", "EPS1DOT", "EPS2DOT"]
# only for ELL1 using FB0
_ell1fbextraparnames = [
    "M2",
    "SINI",
    "A1DOT",
    "EPS1DOT",
    "EPS2DOT",
]
# only for ELL1H
_ell1hextraparnames = ["PBDOT", "A1DOT", "EPS1DOT", "EPS2DOT", "H3", "H4"]
# only for DD
_ddextraparnames = ["M2", "SINI", "PBDOT", "A1DOT", "OMDOT", "EDOT"]
# only for DDK
_ddkextraparnames = ["PBDOT", "A1DOT", "OMDOT", "EDOT"]
# if one of these gets added, make sure to add the others too
_linked_pars_to_add = {
    "M2": ["SINI"],
    "SINI": ["M2"],
    "EPS1DOT": ["EPS2DOT"],
    "EPS2DOT": ["EPS1DOT"],
    "H4": ["H3"],
}
# if one of these gets removed, make sure to remove the others too
_linked_pars_to_remove = {
    "M2": ["SINI"],
    "SINI": ["M2"],
    "EPS1DOT": ["EPS2DOT"],
    "EPS2DOT": ["EPS1DOT"],
    "H3": ["H4"],
}
# do not initialize these to 0
_default_values = {"SINI": 0.5, "STIGMA": 0.5}


def astropy_numpy_json_serializer(obj):
    """Serializer to allow dumping of results including quantities and np data to JSON

    Usage
    -----
    >>> s=json.dumps(results,default=parallel_fit.astropy_numpy_json_serializer)

    """
    if isinstance(obj, u.Quantity):
        return {
            "value": (
                obj.value.tolist() if hasattr(obj.value, "tolist") else float(obj.value)
            ),
            "unit": str(obj.unit),
        }
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, (np.float64, np.longdouble)):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def astropy_json_decoder(dct):
    """Decoder to convert JSON back to astropy quantities

    Usage
    -----
    >>> json.loads(s,object_hook=parallel_fit.astropy_json_decoder)

    """
    if "unit" in dct:
        return u.Quantity(dct["value"], unit=dct["unit"])
    return dct


class TestFitter:
    """Class to allow testing of model fits with parameters added/removed.  Fits will be run in parallel.

    Example
    -------
    >>> tf = TestFitter(parfile, timfile)
    >>> results = tf.run(maxiter=20)
    # the results contain meta info about the start and stop time
    # as well as info about each specific parameter examined
    >>> parameters = list(results.keys())
    # the meta info is separate
    >>> parameters.remove('meta')
    # for example
    >>> print(results['PX'])
    # will contain info like:
    # {'number': 0, 'string': 'Combination 0: add PX', '
    # initvalues': [None, 0, 0, 0, 0, 0, 0], 'chi2': np.float64(59.064087888302154), 'dof': np.int16(55),
    # 'results': {'PX': <Quantity 5.20704212 mas>, 'F2': <Quantity 0. Hz / s2>, 'FD1': <Quantity 0. s>,
    # 'FD2': <Quantity 0. s>, 'FD3': <Quantity 0. s>, 'FD4': <Quantity 0. s>, 'FD5': <Quantity 0. s>}}
    # where it lists all of the parameters that have been considered for all of the tests, but all except
    # the parameter of interest should be at their default/null value

    # If you need to adjust the parameter sets, you can do that:
    >>> tf.defaults['extraparnames'].remove('F2')
    >>> tf.setup(maxFD=4)
    # and then re-run the fit

    # Or you can do it like:
    >>> for p in ["PX", "M2", "F2", "SINI", "A1DOT", "EPS1DOT", "EPS2DOT"]:
            tf.extraparnames.remove(p)
    >>> tf.setup()

    # to limit the number of CPUs used:
    >>> results = tf.run(maxiter=20, ncpu=2)

    # There are errors that come from using the parallel execution in a `main` program.  Avoid those like::
    ```
    import multiprocessing
    import pint.logging
    import parallel_fit

    if __name__ == "__main__":
        __spec__ = None
        multiprocessing.freeze_support()

        pint.logging.setup("INFO")
        ...
    ```

    # To run on a `slurm` system where the jobs are distributed to different nodes::
    ```
    import multiprocessing
    from pathlib import Path
    import pint
    from pint.models import get_model_and_toas
    import pint.logging
    import sys
    import datetime
    import json
    import cfut
    import concurrent.futures
    import parallel_fit

    # these determine the node type, outout destination
    # and also the specific conda environment to load
    # customize as needed
    slurm_options = ["#SBATCH --cpus-per-task=1",
                    "#SBATCH --mem=32g",
                    "#SBATCH --partition=256g",
                    "#SBATCH --output=slurm/slurm-%j-%n-%N.out",
                    "#SBATCH --error=slurm/slurm-%j-%n-%N.err",
                    "source /sharedapps/LS/cgca/conda/bin/activate pintdev_312"]

    def main():
        pint.logging.setup("DEBUG")

        log.info(f"Starting at {datetime.datetime.now().isoformat()}")
        print(f"Starting at {datetime.datetime.now().isoformat()}")

        multiprocessing.freeze_support()


        m, t = get_model_and_toas(...)
        tf = parallel_fit.TestFitter(model=m, toas=t)
        # use this for SLURM
        with cfut.SlurmExecutor(debug=True,keep_logs=True,additional_setup_lines=slurm_options) as executor:
            results=tf.run(maxiter=20, executor=executor, printprogress=False)
            print(
                f"Finished at {datetime.datetime.now().isoformat()}"
                )
            log.info(
                f"Finished at {datetime.datetime.now().isoformat()}"
                )

            with open(..., "w") as fo:
                json.dump(
                    results,
                    fo,
                    indent=4,
                    default=parallel_fit.astropy_numpy_json_serializer,
                )

    if __name__ == "__main__":
        multiprocessing.freeze_support()
        main()
    ```
    # Note that I found this worked with `cfut==0.4` and not the latest version.
    # The above command should be saved to a file and the run with `srun` or `sbatch`.  The following worked
    # but is also probably redundant::
    ```
    #!/bin/bash -l
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=16g
    #SBATCH --output=slurm/slurm-%j.out
    #SBATCH --error=slurm/slurm-%j.err

    # use pintdev environment to use gridding updates
    conda activate pintdev_312

    echo "Starting"
    date
    srun -u python ...
    echo "Done"
    date
    ```
    """

    def __init__(
        self,
        parfile: Optional[file_like] = None,
        timfile: Optional[file_like] = None,
        model: Optional[pint.models.TimingModel] = None,
        toas: Optional[pint.toa.TOAs] = None,
        maxFB: int = 5,
        maxFD: int = 5,
    ):
        """
        Parameters
        ----------
        parfile : str, optional
            The parfile name, or a file-like object to read the parfile contents from
        timfile : str, optional
            The timfile name, or a file-like object to read the timfile contents from
        model : pint.models.TimingModel, optional
            An existing timing model if ``parfile`` is not supplied
        toas : pint.toa.TOAs, optional
            An existing set of TOAs if ``toas`` is not specified
        maxFB: int, optional
            Maximum FB number to include
        maxFD: int, optional
            Maximum FD number to include
        """
        if parfile is not None and timfile is not None:
            self.parfile = parfile
            self.timfile = timfile
            self.m_base = get_model(self.parfile)
        elif model is not None and toas is not None:
            self.m_base = copy.deepcopy(model)
            self.parfile = io.StringIO(model.as_parfile())
            self.timfile = None
            self.toas = toas
        else:
            raise ValueError("Must supply either parfile/timfile or model/toas")
        self.defaults = {}
        self.set_defaults()
        self.extrapars = None
        # load in the base model first to see which parameters needs to be added vs removed
        self.setup(maxFB=maxFB, maxFD=maxFD)

    def setextrapars(
        self,
        maxFB: int = 5,
        maxFD: int = 5,
    ):
        """
        Define the extra parameters to test.

        If the lists of parameters change (e.g., from an external file) this can be re-run

        Parameters
        ----------
        maxFB: int, optional
            Maximum FB number to include
        maxFD: int, optional
            Maximum FD number to include

        """
        self.extraparnames = self.defaults["extraparnames"].copy()
        ell1fbextraparnames = self.defaults["ell1fbextraparnames"].copy()
        for j in range(1, maxFB + 1):
            ell1fbextraparnames.append(f"FB{j}")
        if "BINARY" in self.m_base.params:
            if self.m_base["BINARY"].value == "ELL1":
                if "PB" in self.m_base.params and self.m_base["PB"].value is not None:
                    self.extraparnames += self.defaults["ell1pbextraparnames"]
                if "FB0" in self.m_base.params and self.m_base["FB0"].value is not None:
                    self.extraparnames += ell1fbextraparnames
            elif self.m_base["BINARY"].value == "ELL1H":
                self.extraparnames += self.defaults["ell1hextraparnames"]
            elif self.m_base["BINARY"].value == "DD":
                self.extraparnames += self.defaults["ddextraparnames"]
            elif self.m_base["BINARY"].value == "DDK":
                self.extraparnames += self.defaults["ddkextraparnames"]
        for j in range(1, maxFD + 1):
            self.extraparnames.append(f"FD{j}")

    def setup(self, maxFB: int = 5, maxFD: int = 5, baseline: bool = True):
        """
        Set up the correct combinations of tests based on the loaded timing model and the current lists of parameters

        If the lists of parameters change (e.g., from an external file) this can be re-run

        Parameters
        ----------
        maxFB: int, optional
            Maximum FB number to include
        maxFD: int, optional
            Maximum FD number to include
        baseline: bool, optional
            Whether or not to include the baseline model (with no parameters added or removed)

        """
        if self.extrapars is None:
            self.setextrapars(maxFB=maxFB, maxFD=maxFD)
        default_values = self.defaults["default_values"].copy()

        # which actually have to be added to create the maximal model
        # because they are not in the model or are in but UNSET
        self.extrapars = {
            x: default_values.get(x, 0)
            for x in self.extraparnames
            if not x in self.m_base.params or self.m_base[x].value is None
        }
        if self.timfile is not None:
            self.m, self.t = get_model_and_toas(
                self.parfile, self.timfile, **self.extrapars
            )
        else:
            self.m = get_model(self.parfile, **self.extrapars)
            # rewind to preserve for future reads
            if isinstance(self.parfile, io.StringIO):
                self.parfile.seek(0)
            self.t = copy.deepcopy(self.toas)
        self.addpars = []
        self.removepars = []
        for p in self.extraparnames:
            if (
                not p in self.m_base.params
                or self.m_base[p].value is None
                or self.m_base[p].value == 0
            ):
                self.addpars.append(p)
                log.info(f"Testing addition of {p}")
            else:
                self.removepars.append(p)
                log.info(f"Testing removal of {p}")

        self.f0 = pint.fitter.Fitter.auto(self.t, self.m)
        self.linked_pars_to_add = self.defaults["linked_pars_to_add"].copy()
        self.linked_pars_to_remove = self.defaults["linked_pars_to_remove"].copy()
        # make sure that if we add FDn, we have all of FDn-1
        for j in range(2, maxFD + 1):
            self.linked_pars_to_add[f"FD{j}"] = [f"FD{i}" for i in range(1, j)]
        # make sure that if we add FBn, we have all of FBn-1
        for j in range(2, maxFB + 1):
            self.linked_pars_to_add[f"FB{j}"] = [f"FB{i}" for i in range(1, j)]
        # if we remove FDn, make sure we also remove all of FDn+1
        for j in range(1, maxFD):
            self.linked_pars_to_remove[f"FD{j}"] = [
                f"FD{i}" for i in range(j + 1, maxFD + 1)
            ]
        # if we remove FBn, make sure we also remove all of FBn+1
        for j in range(1, maxFB):
            self.linked_pars_to_remove[f"FB{j}"] = [
                f"FB{i}" for i in range(j + 1, maxFB + 1)
            ]
        self.parnames = self.addpars + self.removepars
        self.parvalues = []
        combination_number = 0
        self.testinfo = {"meta": {"starttime": Time.now().iso}}

        # start with the baseline model
        if baseline:
            combination = []
            for p in self.addpars:
                combination.append(0)
            for p in self.removepars:
                combination.append(None)
            self.parvalues.append(combination)
            self.testinfo["baseline"] = {
                "number": combination_number,
                "string": (f"Combination {combination_number}: default"),
                "initvalues": combination,
            }
            log.debug(self.testinfo["baseline"]["string"])
            combination_number += 1

        for addpar in self.addpars:
            message_string = []
            combination = []
            for p in self.addpars:
                # for all pars except the one we want to test, make it 0
                if p == addpar:
                    message_string.append(f"{p}")
                    combination.append(None)
                else:
                    if (
                        addpar in self.linked_pars_to_add
                        and p in self.linked_pars_to_add[addpar]
                    ):
                        message_string.append(f"{p}")
                        combination.append(None)
                    else:
                        combination.append(0)
            for p in self.removepars:
                # keep these free by default
                combination.append(None)
            if combination not in self.parvalues:
                # do not duplicate tests
                self.parvalues.append(combination)
                self.testinfo[addpar] = {
                    "number": combination_number,
                    "string": (
                        f"Combination {combination_number}: add {' and '.join(message_string)}"
                    ),
                    "initvalues": combination,
                }
                log.debug(self.testinfo[addpar]["string"])
                combination_number += 1

        for removepar in self.removepars:
            message_string = []
            combination = []
            for p in self.addpars:
                # keep these out by default
                combination.append(0)
            for p in self.removepars:
                # for all pars except the one we want to test, make it None
                if p == removepar:
                    message_string.append(f"{p}")
                    combination.append(0)
                else:
                    if (
                        removepar in self.linked_pars_to_remove
                        and p in self.linked_pars_to_remove[removepar]
                    ):
                        message_string.append(f"{p}")
                        combination.append(0)
                    else:
                        combination.append(None)
            if combination not in self.parvalues:
                # do not duplicate tests
                self.parvalues.append(combination)
                self.testinfo[removepar] = {
                    "number": combination_number,
                    "string": (
                        f"Combination {combination_number}: remove {' and '.join(message_string)}"
                    ),
                    "initvalues": combination,
                }
                log.debug(self.testinfo[removepar]["string"])
                combination_number += 1

    def set_defaults(self):
        """Update the default parameter sets

        This can be overridden elsewhere
        """
        self.defaults["extraparnames"] = _extraparnames
        self.defaults["ell1pbextraparnames"] = _ell1pbextraparnames
        self.defaults["ell1fbextraparnames"] = _ell1fbextraparnames
        self.defaults["ell1hextraparnames"] = _ell1hextraparnames
        self.defaults["ddextraparnames"] = _ddextraparnames
        self.defaults["ddkextraparnames"] = _ddkextraparnames
        self.defaults["linked_pars_to_add"] = _linked_pars_to_add
        self.defaults["linked_pars_to_remove"] = _linked_pars_to_remove
        self.defaults["default_values"] = _default_values

    def run(
        self,
        executor: Optional[concurrent.futures.Executor] = None,
        ncpu: Optional[int] = None,
        chunksize: int = 1,
        printprogress: bool = True,
        **fitargs,
    ) -> Dict:
        """
        Run the fits in parallel using :func:`pint.gridutils.tuple_chisq`

        Parameters
        ----------
        executor : concurrent.futures.Executor or None, optional
                Executor object to run multiple processes in parallel
                If None, will use default :class:`concurrent.futures.ProcessPoolExecutor`, unless overridden by ``ncpu=1``
        ncpu : int, optional
            If an existing Executor is not supplied, one will be created with this number of workers.
            If 1, will run single-processor version
            If None, will use :func:`multiprocessing.cpu_count`
        chunksize : int
            Size of the chunks for :class:`concurrent.futures.ProcessPoolExecutor` parallel execution.
            Ignored for :class:`concurrent.futures.ThreadPoolExecutor`
        printprogress : bool, optional
            Print indications of progress (requires :mod:`tqdm` for `ncpu`>1)
        fitargs :
            additional arguments pass to fit_toas()

        Returns
        -------
        testinfo : dict
            Contains info about test for each individual parameter (plus additional ``meta`` info):
            - 'number' (int): number of test run
            - 'string' (str): description of test
            - 'initvalues' (list): initial values for each parameter
            - 'chi2' (float): final fitted chi^2
            - 'dof' (int): final degrees of freedom
            - 'results': dict with the best-fit vaue for the parameters that are fitted, and 0 when not

        Notes
        -----
        This uses :func:`pint.gridutils.tuple_chisq` to do the fitting.
        By default this will create a :class:`~concurrent.futures.ProcessPoolExecutor` instance
        with ``max_workers`` equal to the desired number of cpus.
        However, if you are running this as a script you may need something like::

            import multiprocessing

            if __name__ == "__main__":
                multiprocessing.freeze_support()
                ...
                grid_chisq(...)

        If an instantiated :class:`~concurrent.futures.Executor` is passed instead, it will be used as-is.

        The behavior for different combinations of `executor` and `ncpu` is:
        +-----------------+--------+------------------------+
        | `executor`      | `ncpu` | result                 |
        +=================+========+========================+
        | existing object | any    | uses existing executor |
        +-----------------+--------+------------------------+
        | None	      | 1      | uses single-processor  |
        +-----------------+--------+------------------------+
        | None	      | None   | creates default        |
        |                 |        | executor with          |
        |                 |        | ``cpu_count`` workers  |
        +-----------------+--------+------------------------+
        | None	      | >1     | creates default        |
        |                 |        | executor with desired  |
        |                 |        | number of workers      |
        +-----------------+--------+------------------------+

        Other ``Executors`` can be found for different computing environments:
        * [1]_ for MPI
        * [2]_ for SLURM or Condor

        .. [1] https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor
        .. [2] https://github.com/sampsyo/clusterfutures

        """
        chi2, dof, extra = pint.gridutils.tuple_chisq(
            self.f0,
            self.parnames,
            self.parvalues,
            extraparnames=self.parnames,
            executor=executor,
            ncpu=ncpu,
            chunksize=chunksize,
            printprogress=printprogress,
            **fitargs,
        )
        for p in self.testinfo:
            if p == "meta":
                continue
            j = self.testinfo[p]["number"]
            result_strings = [
                f"{self.parnames[i]}: {extra[self.parnames[i]][j]}"
                for i in range(len(self.parnames))
            ]
            self.testinfo[p]["chi2"] = chi2[j]
            self.testinfo[p]["dof"] = dof[j]
            self.testinfo[p]["results"] = {
                self.parnames[i]: extra[self.parnames[i]][j]
                for i in range(len(self.parnames))
            }
            log.info(
                f"Test {self.testinfo[p]['number']} ({p}): chi^2={chi2[j]:.2f}/{dof[j]}, {', '.join(result_strings)}"
            )
        log.debug(f"extras={extra}")
        self.testinfo["meta"]["stoptime"] = Time.now().iso

        return self.testinfo
