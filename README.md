# Timing analysis

A long-lived repository for NANOGrav Timing analysis work.

Installing on the notebook server
---------------------------------

1. Go to notebook.nanograv.org and sign in with your NANOGrav.org Google Account. Your username should follow the convention of FirstName.LastName@nanograv.org. If you have any issues, please submit a ticket at http://support.nanograv.org and a CyberI team member will address it quickly.

2. Once logged in, you can access the terminal by navigating to the 'New' drop-down and selecting 'Terminal'.

3. Clone timing_analysis to your home directory and checkout the working branch; in a terminal:
```
> cd ~/work/
> git clone git@gitlab.nanograv.org:nano-time/timing_analysis.git
> cd timing_analysis
> git checkout -b 15yr origin/15yr
```

4. Get the latest copy of PINT (eventually, a script will set the environment); in a terminal:
```
> pip install git+git://github.com/nanograv/pint --user
```

5. To install and make sure paths are set up properly, `cd` into `timing_analysis` and:
```
pip install .
```

Current development is being done on the 15yr branch, so please make sure to point merge requests there. Checkout development branches using the naming convention for pulsars/features as follows:
```
> git checkout -b psr/J1234+5678/[initials]
> git checkout -b feature/[brief_description]
```
