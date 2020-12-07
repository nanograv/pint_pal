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


    def add_cell(self, text, mode="code"):
        """
        A helper function to make adding new cells consistent.
        Can be used to make a generic cell as well.

        Parameters
        ==========
        text : block string
        mode (optional) : type of cell to add
        """
        if mode == "code":
            new_cell = new_code_cell
        elif mode == "markdown":
            new_cell = new_markdown_cell
        elif mode == "raw":
            new_cell = new_raw_cell
        self.cells.append(new_cell(source=textwrap.dedent(text)))
    # Convenience functions
    add_code_cell = lambda self, text: self.add_cell(text=text, mode="code")
    add_markdown_cell = lambda self, text: self.add_cell(text=text, mode="markdown")
    add_raw_cell = lambda self, text: self.add_cell(text=text, mode="raw")


    def add_import_cells(self):
        """ Add cells that contains the main imports """
        self.add_markdown_cell('''\
        ### \[imports\] Import packages

        Imports include:

        + Standard Python imports
        + PINT and PINT modules
        + Notebook Utilities\
        ''')
        self.add_code_cell('''\
        import os
        import sys
        import datetime

        # Import PINT
        import pint.toa as toa
        import pint.models as model
        import pint.fitter as fitter
        import pint.observatory as observatory
        import pint.utils as pu

        # Load notebook utilities
        from timingconfiguration import TimingConfiguration
        import plot_utils
        import ftester\
        ''')


    def add_setup_config_cells(self):
        """ Add cells that load the configuration and set up PINT """
        self.add_markdown_cell('''\
        ### \[setup\] Load configuration file, put par and tim into PINT

        These cells will:

        + Load and set definitions from a configuration file
        + Load the par/tim files or the PINT pickle files\
        ''')
        self.add_code_cell('''\
        tc = TimingConfiguration("config.json")
        psr_toas = tc.get_TOAs()
        psr_model = tc.get_model()\
        ''')


    def add_excision_cells(self):
        """ Add cells that perform the various TOA excision stages """
        self.add_markdown_cell('''\
        ### \[excision\] TOA excision

        A dictionary of masked TOAs is created based on the TOA excision database.
        Each step queries the database and the final application is performed at the end.
        ''')
        self.add_code_cell('''\
        mask_dict = dict()

        # Remove TOAs and JUMPs from orphaned receivers
        mask_dict, psr_model = excise_orphan_receivers(mask_dict, psr_model, psr_toas, PSR)

        # Remove TOAs from individually recognized bad epochs
        mask_dict = excise_bad_epochs(mask_dict, psr_toas, PSR)

        # Remove TOAs from various outlier analyses
        mask_dict = excise_outliers(mask_dict, psr_toas, PSR)

        # Apply TOA excision here
        mask_array = np.ones(psr_toas.ntoas ,dtype=np.bool)
        keys = mask_dict.keys()
        for i in range(psr_toas.ntoas):
            if i in keys:
                mask_array[i] = False\
        ''')


    def add_dmx_binning_cells(self):
        """ Add cells that will call the DMX rebinning routines """
        self.add_markdown_cell('''\
        ### \[dmx_fix\] Regenerate DMX bins

        `dmx_fixer()` is run at this stage to regenerate any DMX bins lost by the TOA removal steps above.

        We should also check for ranges where the bandwidth is too low, etc.\
        ''')
        self.add_code_cell('')


    def add_first_fitter_cells(self):
        """ Add cells that perform the first run of the PINT GLS fitter """
        self.add_markdown_cell('''\
        ### \[initial_fit\] First fitter run

        Run the PINT generalized-least-squares (GLS) fitter and calculate the residuals.\
        ''')
        self.add_code_cell('''\
        psr_fitter = getattr(fitter,tc.config['fitter'])(excised_toas, psr_model)
        pint_chi2 = psr_fitter.fit_toas(NITS)
        res = psr_fitter.resids.time_resids.to(u.us).value\
        ''')


    def add_initial_plot_cells(self):
        """
        Add cells that will make a plot of the residuals
        before any noise modeling.
        """
        self.add_markdown_cell('''### \[initial_resids\] Plot initial residuals''')
        self.add_code_cell('''\
        # all residuals vs. time
        plot_utils.plot_residuals_time(excised_toas, res)\
        ''')
        self.add_code_cell('''\
        # Get epoch averaged, whitened, and whitened averaged residuals for plotting
        avg = psr_fitter.resids.ecorr_average(use_noise_model=True)
        wres = ub.whiten_resids(psr_fitter)
        wres_avg = ub.whiten_resids(avg)
        # get rcvr backend combos for averaged residuals
        rcvr_bcknds = np.array(psr_toas.get_flag_value('f')[0])
        avg_rcvr_bcknds = []
        for iis in avg['indices']:
            avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
        avg_rcvr_bcknds = np.array(avg_rcvr_bcknds)\
        ''')
        self.add_code_cell('''\
        # plot averaged residuals v. time
        pup.plot_residuals_time(excised_toas, avg['time_resids'].to(u.us).value, fromPINT = False, \\
                            errs = avg['errors'].value, mjds = avg['mjds'].value, rcvr_bcknds = avg_rcvr_bcknds, \\
                            avg = True)

        # Plot whitened residuals. v. time
        pup.plot_residuals_time(excised_toas, wres.to(u.us).value, figsize=(10,4), fromPINT = True, whitened = True)

        # plot whitened, epoch averaged residuals v. time
        pup.plot_residuals_time(excised_toas, wres_avg.to(u.us).value, fromPINT = False, \\
                            errs = avg['errors'].value, mjds = avg['mjds'].value, rcvr_bcknds = avg_rcvr_bcknds, \\
                            avg = True, whitened = True)\
        ''')


    def add_noise_modeling_cells(self):
        """
        Add a number of cells that will perform noise modeling
        via enterprise
        """
        self.add_markdown_cell('''### \[enterprise_prep\] Prepare `enterprise`''')
        self.add_code_cell('')
        self.add_markdown_cell('''### \[enterprise_run\] Run `enterprise`''')
        self.add_code_cell('')


    def add_residual_stats_cells(self):
        """
        Add cells that will return various statistics
        on the residuals
        """
        self.add_markdown_cell('### \[residual_stats\] Residual statistics')
        self.add_code_cell('')


    def add_Ftest_cells(self):
        """
        Add cells that will perform the various F-test
        parameter significance checks
        """
        self.add_markdown_cell('''### \[F-tests\] Perform F-tests''')
        self.add_code_cell('''\
        Ftest_dict = ftester.run_Ftests(psr_fitter)\
        ''')


    def add_chisq_cells(self):
        """
        Add cell that will report if the reduced chi-squared
        is close to 1.00
        """
        self.add_markdown_cell('''### \[chi-squared\] Check if reduced $\chi^2$ is close to 1.00:''')
        self.add_code_cell('''\
        chi2r = pint_chi2 / psr_toas.ntoas
        if not (0.95 <= chi2r <= 1.05):
            display(Markdown("<span style=\\"color:red\\">Reduced $\chi^2$ is %0.5f / %d = %0.5f</span>"%(pint_chi2, psr_toas.ntoas, chi2r)))
            warnings.warn(r"Reduced $\chi^2$ is far from 1.00")
        else:
            display(Markdown("Reduced $\chi^2$ is %0.5f / %d = %0.5f"%(pint_chi2, psr_toas.ntoas, chi2r)))\
        ''')


    def add_summary_plots_cells(self):
        """
        Add cells that will display various
        summary plots
        """
        self.add_markdown_cell('''### \[plot_residuals\] Plot Residuals''')
        self.add_markdown_cell('''### \[plot_avg_residuals\] Plot Epoch-Averaged Residuals''')
        self.add_markdown_cell('''### \[plot_residuals\] Plot DMX''')


    def add_output_dmx_cells(self):
        """ Add cells to run DMX parser from PINT """
        self.add_markdown_cell('''\
        To get the dmxparse file in the tempo version format as a separate file,
        just need to call PINT utils `dmxparse` function. Output file name is `dmxparse.out`
        and will be in the same directory as this notebook.\
        ''')
        self.add_code_cell('''dmx_dict = pu.dmxparse(psr_fitter, save=True)''')


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


    tn.write_out()
