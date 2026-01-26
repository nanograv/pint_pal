import time
from pathlib import Path
import textwrap

import numpy as np
import scipy.stats as stats

# only for software version
import sys
import astropy
import enterprise
import pint_pal

from markdownreport import *

import pint.models as models
from pint.fitter import Fitter
import pint.logging
pint.logging.setup(level="ERROR")

from pint_pal.utils import resid_stats


        
class Summarizer:
    def __init__(self,
                 fitter: pint.fitter.Fitter,
                 parfile: str,
                 fitter_post_noise: pint.fitter.Fitter | None = None,
                 rs_dict: dict | None = None,
                 user: str = "anonymous user",
                 autorun: bool = True,
                 filename: str = "summary.pdf",
                 font_size: int = 10,
                 suppress_errors: bool = False,
                 verbose: bool = False
                 ) -> None:
        
        self.fitter = fitter
        if fitter_post_noise is None: #is this the right behavior?
            print("No post-noise fitter specified; setting post-noise fitter to pre-noise fitter")
            self.fitter_post_noise = fitter
        else:
            self.fitter_post_noise = fitter_post_noise
        self.parfile = parfile
        self.rs_dict = rs_dict

        self.suppress_errors = suppress_errors
        
        self.NB = True
        if fitter.toas.wideband:
            self.NB = False

        self.psr = fitter.model.PSR.value

        self.title = f"{self.psr} {'narrowband' if self.NB else 'wideband'}"
        self.title = self.title.replace("-", "–")

        self.report = MarkdownReport(title=self.title, font_size=font_size)
        

        self.report.add_section_by_title("Errors")




        self.user = user

        if autorun:
            self.generate_residual_stats(threshold=3) #hard coding for now
            self.add_summary_plots()
            self.generate_summary_info()
            self.generate_software_versions()
            self.output_pdf(filename=filename, verbose=verbose)


    def check_error(self, text: str, condition: bool, error: str, anchor: str, new_anchor: bool = True ) -> str:
        """
        Wrap a text string in an appropriate color and link if it is an error to be reported

        Parameters
        ----------
        text : str
            Text to wrap.
        condition : bool
            If condition is True, not an error and continue as normal. Otherwise, generate an error message and highlight appropriately.
        error : str
            Error message to display
        anchor : str
            Name of anchor for error message to link to
        new_anchor: bool, default=True
            If this is a new anchor name or not. If False, it is assumed it is linking to a previously-defined anchor name.
        """
        if condition:
            return text
        else:
            self.report.write(f"[{error}: {text}](#{anchor})", section_title="Errors")
            if new_anchor:
                return color_text(f"<a name=\"{anchor}\">{text}</a>", color="white", highlight="red")
            else:
                return color_text(text, color="white", highlight="red")

            
    def generate_timing_model(self):
        pass

    
    def generate_residual_stats(self, threshold: float = 10) -> None:
        """
        Table of residual stats, originally including the WRMS/RMS of the residuals per band and in total

        Rather than take in the rs_dict from the notebook, the calculation of resid_stats() is redone so that each can be run independently with the full information. But, this is then a little slower each run.

        Parameters
        ----------
        threshold : float, default=10
            RMS threshold to raise a warning, in microseconds.
        """

        self.report.add_section_by_title("Residual Statistics")
        table_data = [("Parameter", "Value 0", "Value 1")]

        if self.rs_dict is None:
            dm_dict = None
            epoch_avg = 'ecorr_noise' in self.fitter_post_noise.model.get_components_by_category()
            if self.NB:
                rs_dict = resid_stats(self.fitter_post_noise, 
                                      epoch_avg=epoch_avg,
                                      whitened=True, 
                                      print_pretty=False)
            else:
                rs_dict, dm_dict = resid_stats(self.fitter_post_noise, 
                                               whitened=True, 
                                               dm_stats=True, 
                                               print_pretty=False)
        else:
            rs_dict = self.rs_dict


        rs_keys = rs_dict.keys()
        table = textwrap.dedent(
            """
            | <a name="rms_table">Statistic</a> | Frontend/Backend | Value |
            | --------- | ---------------- | ----- |
            """
        )
        for key in rs_keys:
            for mode in ["wrms", "rms"]:

                value = self.check_error(f"{rs_dict[key][mode].value:.3f} {rs_dict[key][mode].unit}", (rs_dict[key][mode].value < threshold), f"Large {mode.upper()} found ({key})", "rms_table", new_anchor=False)
                table += f"| {mode.upper()} | {key} | {value} |\n"
                
        self.report.write(table)

        # Reduced chi-squared calculations
        chi2_0 = self.fitter.resids_init.chi2
        ndof_0 = self.fitter.resids_init.dof
        rchi2_0 = chi2_0/ndof_0
        fpp_0 = stats.chi2(int(ndof_0)).sf(float(chi2_0))

        rchi2_0_text = self.check_error(f"{rchi2_0:0.6f}", (0.95 < rchi2_0 < 1.05), "Large reduced χ2 found", "rchi2_0")

        fpp_0_text = self.check_error(f"{fpp_0:0.3f}", (0.001 < fpp_0 < 0.999), "Bad residual false positive probability found", "fpp_0")

        self.report.write("Before applying new noise model:")
        self.report.write(f"Reduced χ2 is {chi2_0:0.6f}/{ndof_0} = {rchi2_0_text} (false positive probability = {fpp_0_text})")


        if self.fitter_post_noise is not None:
            self.report.write("After applying new noise model:")
            chi2_1 = self.fitter_post_noise.resids_init.chi2
            ndof_1 = self.fitter_post_noise.resids_init.dof
            rchi2_1 = chi2_1/ndof_1
            fpp_1 = stats.chi2(int(ndof_1)).sf(float(chi2_1))

            rchi2_1_text = self.check_error(f"{rchi2_1:0.6f}", (0.95 < rchi2_1 < 1.05), "Large reduced χ2 found", "rchi2_1")
            
            fpp_1_text = self.check_error(f"{fpp_1:0.3f}", (0.001 < fpp_1 < 0.999), "Bad residual false positive probability found", "fpp_1")

            self.report.write(f"Reduced χ2 is {chi2_1:0.6f}/{ndof_1} = {rchi2_1_text} (false positive probability = {fpp_1_text})")
        else:
            self.report.write("No post-noise model found.") #this does not work
        return


    def generate_residual_plots(self) -> None:
        pass


    def generate_summary_info(self) -> None:


        self.report.add_section_by_title("Summary Information")
        

        # Print timestamp
        timestamp = time.strftime("%Y %b %d (%a) %H:%M:%S GMT", time.gmtime())
        self.report.write(f"Summary generated on {timestamp} by {self.user}\n")
        # Print input par file
        self.report.write(f"Input par file: `{self.parfile}`\n")
        # Print input tim information
        filenames = self.fitter.toas.filename
        if isinstance(filenames, str):
            tim_filenames = [filenames]
        else:
            tim_filenames = filenames
        self.report.write(f"Input tim file directory: `{str(Path(tim_filenames[0]).resolve().parent)}`\n")
        self.report.write(f"Input tim files:\n")

        #text =  "<b><ul>" + "".join([f"<li>{filename}</li>" for filename in tim_filenames]) + "</ul></b><br>"
        text = "".join([f"+ `{filename}`\n" for filename in tim_filenames])
        self.report.write(text)

        # Print span
        start = self.fitter.toas.first_MJD
        finish = self.fitter.toas.last_MJD
        start_ymd = start.to_value(format='iso')
        finish_ymd = finish.to_value(format='iso')
        span = (finish.value - start.value)/365.25


        self.report.write(f"Span: {span:.1f} years ({str(start_ymd).split(' ')[0]} – {str(finish_ymd).split(' ')[0]})\n")

        return

    def generate_software_versions(self) -> None:
        """
        Write out the software versions used
        """

        self.report.add_section_by_title("Software versions used in timing analysis", remove_spacing=True)

        self.report.write(f"PINT: `{pint.__version__}`")
        self.report.write(f"pint_pal: `{pint_pal.__version__}`")
        self.report.write(f"astropy: `{astropy.__version__}`")
        self.report.write(f"numpy: `{np.__version__}`")
        self.report.write(f"python: `{sys.version}`")
        self.report.write(f"enterprise: `{enterprise.__version__}`")



    def add_summary_plots(self) -> None:
        """
        Rather than generating each set of plots with a call here, we will use the plot_utils functionality for now.
        """
        self.report.add_section_by_title("Plots")
        self.report.write(f"![Residuals and DMX vs time]({self.psr}_summary_plot_1_nb.png)")
        self.report.write(f"![Whitened residuals vs time and residual statistics]({self.psr}_summary_plot_2_nb.png)")
        self.report.write(f"![Whitened residuals/uncertainty vs time and residual/uncertainty statistics]({self.psr}_summary_plot_3_nb.png)")
        self.report.write(f"![Residual vs frequency]({self.psr}_summary_plot_4_nb.png)")
        return


    
    def output_pdf(self, filename: str = 'summary.pdf', verbose: bool = False) -> None:
        """
        Create the PDF.

        Parameters
        ----------
        filename : str, default="summary.pdf"
            Name of the output PDF
        verbose : bool, default=False
            If True, prints the commands generated and the HTML header. Passes to the MarkdownReport.
        """
        if self.suppress_errors or len(self.report.get_section("Errors").content) == 0:
            self.report.delete_section_by_title("Errors")
            
        self.report.generate_pdf(filename, verbose = verbose)




