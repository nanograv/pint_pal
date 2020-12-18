"""
This is the primary script containing the "recipe" that will
generate a NANOGrav-specific pipeline notebook using the TimingNotebook class.

Very basic usage:
> python notebook_writer.py config/J1910+1256.yaml
This will output a notebook named process.ipynb that runs on the J1910+1256.yaml
configuration file in the config/ directory.
"""

import argparse
from timingnotebook import TimingNotebook


## Argument parser setup
parser = argparse.ArgumentParser(description="""\
                                 NANOGrav Notebook Writer
                                 Outputs a .ipynb with the appropriate variables set
                                 """)
parser.add_argument("config", \
                    type=str, help="YAML configuration filename")
parser.add_argument("-f", "--filename", default="process.ipynb", \
                    type=str, help="Output filename")
parser.add_argument("-w", "--working", action="store_true", \
                    help="Write out the working notebook template used \
                    for working through the timing models.")
parser.add_argument("-t", "--timdir", default=None, \
                    type=str, help="Path to directory with tim file")
parser.add_argument("-p", "--pardir", default=None, \
                    type=str, help="Path to directory with par file")

args = parser.parse_args()


tn = TimingNotebook()
if args.working:
    tn.add_cell('Preliminary Timing Notebook\n=====', mode="markdown")
else:
    tn.add_cell('Processing Notebook\n=====', mode="markdown")
#tn.add_markdown_cell('''\
#The notebook produced here is a test of the capabilities of the notebook
#writer. For convenience in discussion, tags are provided in brackets.\
#''')
tn.add_import_cells()

tn.add_debug_setup_cells()
tn.add_setup_config_cells(filename=args.config, tim_directory=args.timdir, \
                          par_directory=args.pardir)

tn.add_markdown_cell('''\
---

# TOA and Model Preparation Stage

---\
''')

#tn.add_excision_cells()
tn.add_dmx_binning_cells()
tn.add_first_fitter_cells()
tn.add_initial_plot_cells()

if args.working:
    tn.add_fit_testing_cells()

else:
    tn.add_markdown_cell('''\
    ---

    # Noise Modeling Stage

    ---\
    ''')
    tn.add_noise_modeling_cells()

    tn.add_markdown_cell('''\
    ---

    # Finalize Timing Solutions

    ---\
    ''')
    tn.add_residual_stats_cells()
    tn.add_Ftest_cells()
    tn.add_chisq_cells()

    tn.add_markdown_cell('''\
    ---

    # Make Summary Plots

    ---\
    ''')
    tn.add_summary_plots_cells()

    tn.add_markdown_cell('''\
    ---

    # Output Files

    ---\
    ''')
    tn.add_output_dmx_cells()

tn.write_out(filename=args.filename)
