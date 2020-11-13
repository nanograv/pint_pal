"""
This is the primary script containing the "recipe" that will
generate a NANOGrav-specific pipeline notebook using the TimingNotebook class.
"""

from timing_analysis.timingnotebook import TimingNotebook



tn = TimingNotebook()
tn.add_cell('Outline Processing Notebook\n=====', mode="markdown")
tn.add_markdown_cell('''\
The notebook produced here is a test of the capabilities of the notebook
writer. For convenience in discussion, tags are provided in brackets.\
''')
tn.add_import_cells()

tn.add_setup_config_cells()

tn.add_markdown_cell('''\
---

# TOA and Model Preparation Stage

---\
''')

tn.add_excision_cells()
tn.add_dmx_binning_cells()
tn.add_first_fitter_cells()
tn.add_initial_plot_cells()

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

tn.write_out()
