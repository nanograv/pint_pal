This directory contains scripts and generated data (in .pickle files) to generate tables for the 15yr data paper.

Much of what is in here was heavily modified by Scott Ransom from the 12.5yr table scripts, updating things to use PINT and ApJ's DeluxeTable environment.

Some comments:
  - `make_par_table.py`, `make_model_table.py`, and `get_residuals.py` need a link (or a copy) of the proper release path (meaning directory with par/tim files) set (i.e. `release_path` in those scripts)
  - `good_TOAs.txt` is the path to the above directory as it would be seen on the Notebook server
  - The .pickle files were generated using `get_residuals.py` running locally, and `read_noise_chains.py` on the Notebook server
  - `utils.py` are several useful functions pulled from timing_analysis to allow these scripts to run without the full, complicated, 15yr timing environment being present
  - `test.tex` is a simple LaTeX Doc that you can use to view the tables you generate (and debug them)

Scott
