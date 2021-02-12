"""
This is the primary script containing the "recipe" that will
generate a NANOGrav-specific pipeline notebook using the TimingNotebook class.

Very basic usage:
> python notebook_writer.py config/J1910+1256.nb.yaml
This will output a notebook named process.ipynb that runs on the J1910+1256.nb.yaml
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
parser.add_argument("-t", "--timdir", default=None, \
                    type=str, help="Path to directory with tim file")
parser.add_argument("-p", "--pardir", default=None, \
                    type=str, help="Path to directory with par file")
parser.add_argument("-a", "--autorun", action="store_true", \
                    help="For batch runs; removes markdown/plotting/interactive cells")
parser.add_argument("--use_existing_noise",action="store_true", \
                    help="Use existing noise analysis results")
parser.add_argument("--run_noise",action="store_true", \
                    help="Run noise analysis")
parser.add_argument("-w", "--write_results",action="store_true", \
                    help="Write resulting par/tim/dmx files")
parser.add_argument("--prenoise_only",action="store_true", \
                    help="Lite notebook with setup/prenoise only")
args = parser.parse_args()


tn = TimingNotebook()


tn.add_setup(autorun=args.autorun)
tn.add_prenoise(filename=args.config,tim_directory=args.timdir,
                par_directory=args.pardir,write=args.write_results,
                autorun=args.autorun)

if not args.prenoise_only:
    tn.add_noise(use_existing=args.use_existing_noise,run_noise=args.run_noise,
                 write=args.write_results,autorun=args.autorun)
    tn.add_compare(autorun=args.autorun)
    tn.add_significance(autorun=args.autorun)
    tn.add_summary(autorun=args.autorun)
    tn.add_changelog(autorun=args.autorun)


tn.write_out(filename=args.filename)
