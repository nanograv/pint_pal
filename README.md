# Timing analysis

A long-lived repository for NANOGrav Timing analysis work.

Installing on the notebook server
---------------------------------

1. Go to notebook.nanograv.org and sign in with your NANOGrav.org Google Account. Your username should follow the convention of FirstName.LastName@nanograv.org. If you have any issues, please submit a ticket at http://support.nanograv.org and a CyberI team member will address it quickly.

2. Once logged in, you can access the terminal by navigating to the 'New' drop-down and selecting 'Terminal'. (Note: you will be user "jovyan" but in your own separate userspace/container)

3. Clone timing_analysis to your (i.e. "jovyan") home directory and checkout the working branch; in a terminal:
```
> cd ~/work/
> git clone git@gitlab.nanograv.org:nano-time/timing_analysis.git
> cd timing_analysis
> git checkout -b 15yr origin/15yr
```
If the last command tells you `15yr` exists already, great.

In order to ensure commits are pushed as your NANOGrav.org GitLab account rather than the "jovyan" user, please run the following commands to configure the `timing_analysis` directory.
```
cd ~/work/timing_analysis/
git config user.name "FirstName LastName"
git config user.email "FirstName.LastName@nanograv.org"
```
Note: You may have to reconfigure this if your container is brought down at any point. This should be remedied in the future.

4. Get the latest copy of PINT; in a terminal:
```
> pip install git+git://github.com/nanograv/pint --user --upgrade
```
Even if PINT is already installed, run this same command to update to the latest version.

5. To install and make sure paths are set up properly, `cd` into `timing_analysis` and:
```
pip install -e .
```
If it is already installed, then `cd` into `timing_analysis`, run `git pull` and then `pip install .`


Timing workflow
---------------

This package has a variety of tools to support timing for NANOGrav, but the basic goal here is to produce a config `.yaml` file and a `.par` file that together produce clean timing residuals for a new pulsar. Note that for each pulsar, there will be wideband (wb) and narrowband (nb) versions of both configuration and parameter files. For examples that follow, we will use narrowband file naming conventions and we recommend using these files for now. (If the pulsar is new to long-term timing, you may need more tools than this to put together an initial `.par` file.) This section will describe how to do that.

1. Pick a pulsar for which timing hasn't been finalized, but for which `.tim` and initial `.par` files exist. The easiest may just be to look in `results` and the most recent `/nanograv/releases/15y/toagen/releases/` for pulsars not represented in `configs`. You might also check https://gitlab.nanograv.org/nano-time/timing_analysis/-/branches to make sure no one is working on it.

2. Make a branch for your work on the new pulsar, say J1234+5678:
```
> git checkout 15yr
> git checkout -b psr/J1234+5678/{your_initials}
```

3. If no .yaml file exists in the configs directory for the pulsar (this really shouldn't be the case), copy `configs/template.nb.yaml` to `configs/J1234+5678.nb.yaml` and fill in the basic parameters, in particular `.par` file (will probably be in `results/`) and `.tim` file(s) (will probably be in the most recent release under `/nanograv/releases/15y/toagen/releases/`). For now you may want to select *narrowband* `.tim` files (indicated by `.nb.tim` rather than `.wb.tim`) and ensure `toa-type` is set correctly in the `.yaml` file. If you are timing a pulsar that's been around for a while, check to see if ASP/GASP `.tim` files are available for your source in the latest release directory and ensure they're listed in the `.yaml` file if so; these were recently added.

4. You may need to select which parameters to fit - at a minimum they should be ones that are in the `.par` file. For position, prefer `ELONG`/`ELAT` rather than `RAJ`/`DECJ` or `LAMBDA`/`BETA`; likewise the proper motion parameters `PMELONG`/`PMELAT`. More, NANOGrav policy is that all pulsars should be fit for at least `ELONG`, `ELAT`, `PMELONG`, `PMELAT`, `PX`, `F0`, `F1` in every pulsar.

5. Copy the template notebook to the root directory (where you should probably work):
```
> cp nb_templates/process_v0.9.ipynb J1234+5678.ipynb
```
Because the notebooks aren't version controlled, you can technically name them whatever you'd like. However, we strongly recommend using the psrname + wb/nb + ipynb formatting (you will likely want to keep nb/wb notebooks separate).

6. Open the notebook, fill in your pulsar name, and try running it. Various things will go wrong.

7. Fix all the things. (See below.)

Timing the pulsar
-----------------

There are a lot of things that can go wrong at this point; that's why this isn't an automated process, and why you have a notebook. Here are a few things that might come up, and some suggestions:

- Can't find your `.par` file: you probably need to use `J1234+5678.12.5yr.par`, that is, don't include the directory.

- Can't find one or all of your `.tim` files: make sure you can see them at that location on the same machine your notebook is running on. You probably need to be on the NANOGrav notebook server.

- Few TOAs and they look weird: Nothing will work if the `toa-type` doesn't match the kind of TOA you're using.

- Tons of PINT `INFO` messages: sadly, this is normal.  However, some of these `INFO` messages arise from `setup_dmx()` in `src/timing_analysis/dmx_utils.py` or from the global settings in `src/timing_analysis/defaults.py`.

- Plots are thrown off by a few points with huge error bars: You should be able to zoom in and identify the MJD, then open the `.tim` file and find which TOAs from this day have huge uncertainties. You can use the excision features in the `.yaml` file to remove these.

- Plots are thrown off by lots of points with huge error bars: consider adjusting `snr-cut` to remove the ones with the worst signal-to-noise.

- Plots are thrown off by a few outliers with normal error bars: For a proper analysis we should definitely understand what's wrong with these. For now it may be okay to just excise them. It's not as easy as one might wish to figure out which ones they are yet, but there is the function `large_residuals()` which will print out some of the largest in a format suitable for excision.

- Post-fit model is bad: You may need to fit for more parameters, or switch timing models. 

- `ELL1` approximation bad warning message: You probably want to switch to `DD`; they have suggested parameters there, so just create a temporary `.par` file and edit it, putting those values in. This may require some fitting. 

- The fit is so bad you have a phase wrap: You may be able to resolve this by temporarily using the `.yaml` to restrict to a shorter time interval, or using PINT commands to do so, fit, write out a `.par` file with `write_par` and use it for the input, then move to a longer time interval. Repeat as necessary.

- Your `.par` file looks terrible but is fine in TEMPO2: it's probably in the TCB timescale, which PINT does not support. Use `tempo2 -gr transform old.par new.par tdb` to convert it.

- You see weird kinks in the time series, or peaks at orbital phase 0.25, or sinusoidal variations with orbital phase that change over time: you may need to add features to your timing model.

There are a number of helpful functions available in `timing_analysis.lite_utils` that are imported at the top of the template notebook; you may want to look inside the `src/timing_analysis/lite_utils.py`, which contains the functions and some documentation. You might also try `import timing_analysis.lite_utils; dir(lite_utils)`.
There are also global default settings in `src/timing_analysis/defaults.py`.

Submitting a good timing solution
---------------------------------

When you have a post-fit timing solution that seems good - no wild outliers, no visible structure in terms of time or orbital phase, reduced chi-squared not too far above 1, no warnings from the timing model - you are probably ready to commit the new timing model to the `timing_analysis` repository. In steps summarized below, we use a narrowband files as examples.

1. Generate an output `.par` file; this can be done by having the notebook run `write_par(fo,toatype=tc.get_toa_type())` on a successful fitter `fo`. This will create a file `J1234+5678_PINT_YYYYMMDD.nb.par` in your working directory if you have the `toa-type` field in your `.yaml` file set to `NB`.

2. Archive the old `.par` file and put the new one in place (note: the initial `.par` file naming convention may vary):
```
> git mv results/J1234+5678.12.5yr.par results/archive/
> cp J1234+5678_PINT_YYYYMMDD.nb.par results/
> git add results/J1234+5678_PINT_YYYYMMDD.nb.par
```

3. Update `configs/J1234+5678.nb.yaml` (or `configs/J1234+5678.nb.yaml`) to use this new par file; place the old one as `compare-model`. Rerun the notebook and confirm that all is well and that both pre-fit and post-fit are good fits, and indistinguishable.

4. Go through the checklist at https://gitlab.nanograv.org/nano-time/timing_analysis/-/wikis/Review-Checklists-For-Merging

5. Submit it to the gitlab:
```
> git commit
> git push
```
An error message appears with the command you should have run instead; run that. (It'll be something involving `--set-upstream origin` but it's easier to just let git suggest it.)

6. Create a merge request for your branch - this asks one of the maintainers to look at your work and if it's okay make it part of the official repository. There are two ways to do this -- you can copy and paste a link that appears in the terminal output when you push your changes, or use the "new merge request" button [here](https://gitlab.nanograv.org/nano-time/timing_analysis/merge_requests) (see the screenshot below). If you don't see the button, you may need to be added as a developer (check with Joe Swiggum or Joe Glaser). You can include additional information in your MR with the description and comments. Attaching the summary PDF is recommended.

![screenshot](new-merge-request.png)

7. Respond to any comments or questions or requests for adjustment the maintainers raise; when they are happy they will merge it.

8. Document your (significant) changes in the yaml! Each yaml file has a "changelog" section at the bottom. It's expecting entries in this format:  - ['YYYY-MM-DD user.name KEYWORD: (explain yourself here)']. There is a function in lite_utils called new_changelog_entry that can help you format these entries. The template nb also has a cell explaining change logging.   

It's probably also worth checking the [high-level decisions](https://gitlab.nanograv.org/nano-time/timing_analysis/-/wikis/High-level-decisions-for-v0).

Noise modeling
---------------

Make sure to run the following commands _before_ attempting noise modeling (you might have to restart your kernel if it's already running):
```
> pip install git+https://github.com/nanograv/enterprise.git --upgrade --user
> pip install git+https://github.com/nanograv/enterprise_extensions.git --upgrade --user
```

Noise modeling in the notebook is implemented through the use of the `run_noise_analysis` flag. This flag is set to __False__ by default. __Do not__ perform noise modeling until everything else for the pulsar is finalized, since noise modeling can take a long time to finish, especially for the A-rated and some of the B-rated pulsars. Your workflow should thus look like:

1. Generate a good timing solution for your pulsar _without_ noise modeling.
2. Commit and push the TOAs and parfiles as described above.
3. Re-run the notebook _with_ `run_noise_analysis = True`, i.e. _with_ noise modeling.
4. Commit and push the new noise modeled parfile in the same way as above.

These steps might not work for the new pulsars (C-type) which don't have noise parameters. In these cases, feel free to run the noise analysis since it is quicker than that for A- and B-type pulsars.

Congratulations, you have timed a pulsar for NANOGrav!

Other development
-----------------

Current development is being done on the 15yr branch, so please make sure to point merge requests there. Checkout development branches using the naming convention for pulsars/features as follows:
```
> git checkout -b psr/J1234+5678/[initials]
> git checkout -b feature/[brief_description]
```
