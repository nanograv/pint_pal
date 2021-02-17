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
        """ Add setup helper info, import and log.setLevel cells """
        self.add_markdown_cell_skip('''\
        # \[set-up\], imports

        Reminder (if working on the notebook server): grab current copies of relevant software before doing anything! Make sure your copy of `timing_analysis` is up to date and you're working on a development branch, e.g. `psr/J1234+5678/jks`. Then:
        ```
        > cd ~/work/timing_analysis/
        > pip install git+git://github.com/nanograv/pint --upgrade --user
        > pip install git+https://github.com/nanograv/enterprise.git --upgrade --user
        > pip install git+https://github.com/nanograv/enterprise_extensions.git --upgrade --user
        > pip install .
        ```
        Also, if you haven't done so in a while:
        ```
        > git config user.name "FirstName LastName"
        > git config user.email "first.last@nanograv.org"
        ```\
        ''',autorun)
        self.add_code_cell('''\
        from timing_analysis.lite_utils import *
        from timing_analysis.par_checker import *
        from timing_analysis.plot_utils import *
        from timing_analysis.utils import *
        from timing_analysis.dmx_utils import *
        import timing_analysis.noise_utils as nu
        from timing_analysis.ftester import run_Ftests
        from timing_analysis.timingconfiguration import TimingConfiguration
        import yaml
        from astropy import log
        import pint.fitter
        from pint.utils import dmxparse
        import os
        from astropy.visualization import quantity_support
        quantity_support()

        %matplotlib notebook\
        ''')
        self.add_code_cell_skip('''\
        log.setLevel("INFO") # Set desired verbosity of log statements (DEBUG/INFO/WARNING/ERROR)
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

            using_wideband = tc.get_fitter() == 'WidebandTOAFitter'
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
        Run basic checks for pulsar name, solar system ephemeris, clock correction, tropospheric delays, planet Shapiro delays. Also check for the appropriate number of receiver JUMPs and DMJUMPs and fix them automatically if necessary.\
        ''',autorun)
        self.add_code_cell('''\
        check_settings(mo,to)

        receivers = get_receivers(to)
        add_feJumps(mo,receivers)
        if using_wideband:
            add_feDMJumps(mo,receivers)
        check_jumps(mo,receivers,fitter_type=tc.get_fitter())\
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
                plot_residuals_orb(fo, restype='prefit')\
            ''')
            self.add_markdown_cell('''\
            Check that free parameters follow convention, do the fit, plot post-fit residuals, write a pre-noise par file and print a summary of fit results.\
            ''')
            self.add_code_cell('''\
            fo.model.free_params = tc.get_free_params(fo)
            check_fit(fo,skip_check=tc.skip_check)

            fo.fit_toas()
            plot_residuals_time(fo, restype='postfit')
            if mo.is_binary:
                plot_residuals_orb(fo, restype='postfit')

            fo.print_summary()\
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

        Noise analysis runs are required for the 15-yr v0.9 data set, using the latest available timing model and set of TOAs. For 12.5-yr pulsars, noise runs are currently too computationally expensive to be run on the notebook server. For those pulsars, we will likely conduct runs with acceptable pre-noise solutions and independent HPC resources, then resulting chains will be made available. The exact procedure to do this is still TBD.

        If skipping noise analysis here, set run_noise_analysis = False. Only set this to True for new MSPs once you have an acceptable pre-noise solution and reasonable TOA excision. Or if appropriate results are already available, set use_existing_noise_dir = True to apply these noise parameters to the timing model without re-running model_noise.\
        ''',autorun)

        # Allow these to be toggled in writing the notebook, via argparse
        self.add_code_cell(f'''\
        run_noise_analysis = {run_noise}
        use_existing_noise_dir = {use_existing}\
        ''')
        self.add_markdown_cell_skip('''\
        If `run_noise_analysis = True`, perform noise modeling using enterprise and enterprise_extensions; this cell will likely take at least an hour to run, if not several times that. Status can be monitored once modeling is 1% complete. New noise parameters will be added to the timing model if there are existing results or `model_noise` is run. Redefine the fitter object (`fo`), now with noise parameters, and re-fit.\
        ''',autorun)
        self.add_code_cell_skip('''\
        if run_noise_analysis or use_existing_noise_dir:
            remove_noise(mo)
            nu.model_noise(mo, to, using_wideband = using_wideband, run_noise_analysis = run_noise_analysis)
            mo = nu.add_noise_to_model(mo, using_wideband = using_wideband)

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
        Run F-tests to check significance of existing/new parameters; `alpha` is the F-statistic required for a parameter to be marked as significant. This cell may take 5-10 minutes to run.\
        ''',autorun)
        self.add_code_cell('''\
        Ftest_dict = run_Ftests(fo, alpha = 0.0027)\
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
