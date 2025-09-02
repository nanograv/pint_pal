"""
This code contains the TimingNotebook class, which is used to
generate a pipeline notebook. Each section of cells has been
modularized into different functions.

Very basic usage:
    from timingnotebook import TimingNotebook
    tn = TimingNotebook()
    tn.add_import_cell()
    tn.write_out("process.ipynb")
"""


import os
from base64 import encodestring
import codecs
import textwrap

import nbformat as nbf
from nbformat.v4.nbbase import (
    new_code_cell, new_markdown_cell, new_notebook,
    new_output, new_raw_cell
)


METADATA = {
  "kernelspec": {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python"
  },
  "language_info": {
    "codemirror_mode": {
      "version": 3,
      "name": "ipython"
    },
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
    "version": "3.7.6",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python"
  }
}


class TimingNotebook:
    """
    The class that contains the functionality to add cells,
    store that information, and write to an ipynb file.
    """
    def __init__(self, metadata=METADATA):
        """
        Initialization method

        Parameters
        ==========
        metadata (optional): .ipynb metadata, with the standard defined in METADATA
        """
        self.cells = list()
        self.metadata = metadata


    def add_cell(self, text, mode="code", skip=False):
        """
        A helper function to make adding new cells consistent.
        Can be used to make a generic cell as well.

        Parameters
        ==========
        text : block string
        mode (optional) : type of cell to add
        """
        if not skip:
            if mode == "code":
                new_cell = new_code_cell
            elif mode == "markdown":
                new_cell = new_markdown_cell
            elif mode == "raw":
                new_cell = new_raw_cell
            self.cells.append(new_cell(source=textwrap.dedent(text)))
        else:
            pass

    # Convenience functions
    add_code_cell = lambda self, text: self.add_cell(text=text, mode="code")
    add_markdown_cell = lambda self, text: self.add_cell(text=text, mode="markdown")
    add_raw_cell = lambda self, text: self.add_cell(text=text, mode="raw")

    # skip provides flexibility to skip cells for batch runs (autorun)
    add_code_cell_skip = lambda self, text, skip: self.add_cell(text=text, mode="code", skip=skip)
    add_markdown_cell_skip = lambda self, text, skip: self.add_cell(text=text, mode="markdown", skip=skip)


    def add_setup(self,autorun=False):
        """ Add setup helper info, import, and logging cells """
        self.add_markdown_cell_skip('''\
        # \[set-up\], imports

        Reminder (if working on the notebook server): grab current copies of relevant software before doing anything! Make sure your copy of `pint_pal` is up to date and you're working on a development branch, e.g. `psr/J1234+5678/jks`. Then:
        ```
        > cd ~/work/pint_pal/
        > pip install git+git://github.com/nanograv/pint --upgrade --user
        > pip install git+https://github.com/nanograv/enterprise.git --upgrade --user
        > pip install git+https://github.com/nanograv/enterprise_extensions.git --upgrade --user
        > pip install --user -e .
        ```\
        ''',autorun)
        self.add_code_cell('''\
        from pint_pal.lite_utils import *
        from pint_pal.par_checker import *
        from pint_pal.plot_utils import *
        from pint_pal.utils import *
        from pint_pal.dmx_utils import *
        import pint_pal.noise_utils as nu
        from pint_pal.ftester import run_Ftests
        from pint_pal.timingconfiguration import TimingConfiguration
        import yaml
        from loguru import logger as log
        import pint.fitter
        from pint.utils import dmxparse
        import os, sys
        from astropy.visualization import quantity_support
        quantity_support()

        # notebook gives interactive plots but not until the kernel is done
        %matplotlib notebook
        # inline gives non-interactive plots right away
        #%matplotlib inline\
        ''')
        self.add_code_cell_skip('''\
        LOG_LEVEL = "INFO" # Set desired verbosity of log statements (DEBUG/INFO/WARNING/ERROR)
        log.remove() 
        log.add(sys.stderr, level=LOG_LEVEL) 
        git_config_info()\
        ''',autorun)

    def add_prenoise(self,filename="config.yaml",
                     tim_directory=None, par_directory=None,
                     write=False,autorun=False):
        """ Add cells that load yaml/par/tim and do pre-noise fits """
        self.add_markdown_cell_skip('''\
        # develop/update \[prenoise\] timing solution

        Load configuration (`.yaml`) file, get TOAs and timing model; if you're running from the root of the git distribution, simply edit the `.yaml` file name, otherwise include relevant paths to the `.yaml` file, and `.par`/`.tim` directories as kwargs (see commented example).\
        ''',autorun)

        if filename is not None:
            # Allow for None to be passed if these are not strings.
            if isinstance(filename, str):
                filename = f'"{filename}"'
            if isinstance(tim_directory, str):
                tim_directory = f'"{tim_directory}"'
            if isinstance(par_directory, str):
                par_directory = f'"{par_directory}"'

            self.add_code_cell(f'''\
            tc = TimingConfiguration({filename}, tim_directory={tim_directory}, par_directory={par_directory})

            using_wideband = tc.get_toa_type() == 'WB'
            mo,to = tc.get_model_and_toas()\
            ''')
        else:
            # If no config file is provided, generate template version of notebook
            # Does not yet handle tim_directory/par_directory, but these should not be set with no config.
            self.add_code_cell('''\
            tc = TimingConfiguration("configs/[filename].yaml")
            # tc = TimingConfiguration("[path/to/config/filename].yaml", par_directory="[path/to/results]", tim_directory="[/path/to/wherever/the/tim/files/are]")

            using_wideband = tc.get_fitter() == 'WidebandTOAFitter'
            mo,to = tc.get_model_and_toas()\
            ''')

        self.add_markdown_cell_skip('''\
        Run basic checks for pulsar name, solar system ephemeris, clock correction, ecliptic coordinates, tropospheric delays, planet Shapiro delays, and if applicable, removal of Arecibo data affected by bad LO. Check that TOAs being used are from the latest `toagen` release. Also check for the appropriate number of receiver JUMPs and DMJUMPs and fix them automatically if necessary.\
        ''',autorun)
        self.add_code_cell('''\
        check_settings(mo,to)

        receivers = get_receivers(to)
        add_feJumps(mo,receivers)
        if using_wideband:
            add_feDMJumps(mo,receivers)
        check_jumps(mo,receivers,toa_type=tc.get_toa_type())\
        ''')
        self.add_markdown_cell_skip('''\
        Compute pulse numbers; this ensures that parameter changes made in the model will not break phase connection.\
        ''',autorun)
        self.add_code_cell_skip('''\
        to.compute_pulse_numbers(mo)\
        ''',autorun)

        if not autorun:
            self.add_markdown_cell('''\
            Ensure DMX windows are calculated properly for the current set of TOAs, set non-binary epochs to the center of the data span and print a summary of TOAs included.\
            ''')
            self.add_code_cell('''\
            to = setup_dmx(mo,to,frequency_ratio=tc.get_fratio(),max_delta_t=tc.get_sw_delay())
            center_epochs(mo,to)
            to.print_summary()\
            ''')
            self.add_markdown_cell('''\
            Define the fitter object and inspect prefit residuals vs. MJD (also vs. orbital phase for binary MSPs).\
            ''')
            self.add_code_cell('''\
            fo = tc.construct_fitter(to,mo)
            plot_residuals_time(fo, restype='prefit')
            if mo.is_binary:
                plot_residuals_orb(fo, restype='prefit')
            if using_wideband:
                plot_dm_residuals(fo, restype='prefit')\
            ''')
            self.add_markdown_cell('''\
            Check that free parameters follow convention, do the fit, plot post-fit residuals, write a pre-noise par file and print a summary of fit results.\
            ''')
            self.add_code_cell('''\
            fo.model.free_params = tc.get_free_params(fo)
            check_fit(fo,skip_check=tc.skip_check)\
            ''')
            self.add_code_cell('''\
            fo.fit_toas()
            plot_residuals_time(fo, restype='postfit')
            if mo.is_binary:
                plot_residuals_orb(fo, restype='postfit')
            if using_wideband:
                plot_dm_residuals(fo, restype='postfit')

            fo.print_summary()
            
            chi2_decrease = fo.resids_init.chi2-fo.resids.chi2
            print(f"chi-squared decreased during fit by {chi2_decrease}")
            if hasattr(fo, "converged") and fo.converged:
                print("Fitter has converged")
            else:
                if abs(chi2_decrease)<0.01:
                    print("Fitter has probably converged")
                elif chi2_decrease<0:
                    log.warning("Fitter has increased chi2!")
                else:
                    log.warning("Fitter may not have converged")
            if chi2_decrease > 0.01:
                log.warning("Input par file is not fully fitted")\
            ''')
            self.add_code_cell(f'''\
            write_prenoise = {write}
            if write_prenoise:
                write_par(fo,toatype=tc.get_toa_type(),addext='_prenoise')\
            ''')
        else:
            self.add_code_cell('''\
            to = setup_dmx(mo,to,frequency_ratio=tc.get_fratio(),max_delta_t=tc.get_sw_delay())
            center_epochs(mo,to)
            fo = tc.construct_fitter(to,mo)
            fo.model.free_params = tc.get_free_params(fo)
            check_fit(fo,skip_check=tc.skip_check)
            fo.fit_toas()\
            ''')
            if write:
                self.add_code_cell('''\
                write_par(fo,toatype=tc.get_toa_type(),addext='_prenoise')\
                ''')

    def add_noise(self,run_noise=False,use_existing=False,autorun=False,write=False):
        """ Add a number of cells that will perform noise modeling via enterprise """
        self.add_markdown_cell_skip('''\
        # \[noise\] analysis, re-fit

        Noise analysis runs are required for the 15-yr v0.9 data set, using the latest available timing model and set of TOAs. Once a new `.yaml` file is merged, noise runs will get kicked off on Thorny Flats (HPCs) using the `timing-model` and `toas` indicated therein. For pulsars without existing pre-noise solutions, timers should use the [prenoise] section of this notebook to improve TOA excision, ensure residuals are flat, then submit a merge request with that solution, and wait for noise results to be made available before proceeding.

        We strongly discourage running noise analyses from on the notebook server, since doing so can take several hours (or days!) to complete and hogs lots of shared resources. Set `run_noise_analysis = False` unless you have a good reason to do otherwise.\
        ''',autorun)

        # Allow these to be toggled in writing the notebook, via argparse
        self.add_code_cell(f'''\
        run_noise_analysis = {run_noise}
        use_existing_noise_dir = {use_existing}\
        ''')
        self.add_markdown_cell_skip('''\
        If `run_noise_analysis = True`, perform noise modeling using enterprise and enterprise_extensions; this cell will take a long time to run. Status can be monitored once modeling is 1% complete. New noise parameters will be added to the timing model if there are existing results or `model_noise` is run. Redefine the fitter object (`fo`), now with noise parameters, and re-fit.\
        ''',autorun)
        self.add_code_cell_skip('''\
        if run_noise_analysis or use_existing_noise_dir:
            remove_noise(mo)
            nu.model_noise(mo, to, using_wideband = using_wideband, run_noise_analysis = run_noise_analysis)
            mo = nu.add_noise_to_model(mo, using_wideband = using_wideband, base_dir=tc.get_noise_dir())

            fo = tc.construct_fitter(to,mo)
            fo.model.free_params = tc.get_free_params(fo)

            fo.fit_toas()
            plot_residuals_time(fo, restype='postfit')
            if mo.is_binary:
                plot_residuals_orb(fo, restype='postfit')
            fo.print_summary()\
        ''',autorun)

        # Remove plotting with autorun
        if autorun:
            self.add_code_cell('''\
            nu.model_noise(mo, to, using_wideband = using_wideband, run_noise_analysis = run_noise_analysis)
            if run_noise_analysis or use_existing_noise_dir:
                mo = nu.add_noise_to_model(mo, using_wideband = using_wideband)
                fo = tc.construct_fitter(to,mo)
                fo.model.free_params = tc.get_free_params(fo)
                fo.fit_toas()\
            ''')

        self.add_markdown_cell_skip('''\
        Write resulting `.par` and `.tim` files; also run PINT implementation of dmxparse and save the values in a text file with `save = True` (requires DMX bins in model to run properly).\
        ''',autorun)

        # Allow write to be toggled
        self.add_code_cell(f'''\
        write_results = {write}
        if write_results:
            outpar = None  # None leads to default string value
            write_par(fo,toatype=tc.get_toa_type(),outfile=outpar)
            outtim = None  # None leads to default string value
            write_tim(fo,toatype=tc.get_toa_type(),outfile=outtim)

            dmx_dict = dmxparse(fo, save=True)\
        ''')
        
    def add_compare(self,autorun=False):
        """ Add cells to compare timing models """
        self.add_markdown_cell_skip('''\
        # \[compare\] to previous timing model

        Compare post-fit model to `compare-model` (or pre-fit model, if `compare-model` is not specified in the `.yaml` file). Use `?mo.compare` for more information about verbosity options.\
        ''',autorun)
        self.add_code_cell_skip('''\
        compare_models(fo,model_to_compare=tc.get_compare_model(),verbosity='check',nodmx=True,threshold_sigma=3.)\
        ''',autorun)

    def add_significance(self,autorun=False):
        """ Add cells that calculate resid stats, do F-tests """
        self.add_markdown_cell_skip('''\
        # check parameter \[significance\]

        Get information on the weighted (W)RMS residuals per backend. Set `epoch_avg = True` to get the (W)RMS of the epoch-averaged residuals (does not work for wideband analysis; the timing model must have `ECORR` in order for epoch averaging to work). Set `whitened = True` to get the (W)RMS of the whitened residuals. Set both to `True` to get the (W)RMS of the whitened, epoch-averaged residuals.

        For wideband analysis, set `dm_stats = True` to also return the (W)RMS of the DM residuals.\
        ''',autorun)
        self.add_code_cell('''\
        if not using_wideband:
            rs_dict = resid_stats(fo, epoch_avg = True, whitened = True, print_pretty = True)
        else:
            rs_dict, dm_dict = resid_stats(fo, whitened = True, dm_stats = True, print_pretty = True)\
        ''')
        self.add_markdown_cell_skip('''\
        Run F-tests to check significance of existing/new parameters; `alpha` is the p-value threshold for rejecting the null hypothesis that a parameter is not significant. This cell may take 5-10 minutes to run.\
        ''',autorun)
        self.add_code_cell('''\
        try:
            log.remove()
            log.add(sys.stderr, level="WARNING")
            Ftest_dict = run_Ftests(fo, alpha = 0.0027)
        finally:
            log.remove()
            log.add(sys.stderr, level=LOG_LEVEL)\
        ''')

    def add_summary(self,autorun=False):
        """ Add cells that will generate summary pdfs """
        self.add_markdown_cell_skip('''\
        # generate \[summary\] pdf
        
        Generate summary plots required for pdf summaries. Note: this cell will output white space for the plots, but will save them and incorporate them into the pdf summaries appropriately.\
        ''',autorun)
        self.add_code_cell('''\
        if not using_wideband:
            plots_for_summary_pdf_nb(fo)
        else:
            plots_for_summary_pdf_wb(fo)\
        ''')
        self.add_code_cell('''\
        PARFILE = os.path.join(tc.par_directory,tc.config["timing-model"])
        if not using_wideband:
            pdf_writer(fo, PARFILE, rs_dict, Ftest_dict, append=None)
        else:
            pdf_writer(fo, PARFILE, rs_dict, Ftest_dict, dm_dict = dm_dict)\
        ''')

    def add_changelog(self,autorun=False):
        """ Add cell explaining how to generate changelog entries """
        self.add_markdown_cell_skip('''\
        # \[changelog\] entries

        New changelog entries in the `.yaml` file should follow a specific format and are only added for specified reasons (excising TOAs, adding/removing params, changing binary models, etc.). For more detailed instructions, run `new_changelog_entry?` in a new cell. This function can be used to format your entry, which should be added to the bottom of the appropriate `.yaml` file. Note: make sure your git `user.email` is properly configured, since this field is used to add your name to the entry.\
        ''',autorun)

    def write_out(self, filename="process.ipynb"):
        """
        Write out a new ipython notebook

        Parameters
        ==========
        filename (optional): notebook output filename, default "process.ipynb"
        """
        nb0 = new_notebook(cells=self.cells, metadata=self.metadata)
        f = codecs.open(filename, encoding='utf-8', mode='w')
        nbf.write(nb0, f, 4)
        f.close()
