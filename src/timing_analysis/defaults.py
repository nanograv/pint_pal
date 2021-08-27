# Here we keep track of global default settings

# Choice of clock, SSE
LATEST_BIPM = "BIPM2019"    # latest clock realization to use
LATEST_EPHEM = "DE440"      # latest solar system ephemeris to use

# Toggle various corrections
PLANET_SHAPIRO = True       # correct for Shapiro delay from planets
CORRECT_TROPOSPHERE = True  # correct for tropospheric delays

# DMX model defaults
FREQUENCY_RATIO = 1.1       # set the high/low frequency ratio for DMX bins
MAX_SOLARWIND_DELAY = 0.1   # set the maximum permited 'delay' from SW [us]

# Desired TOA release tag
LATEST_TOA_RELEASE = "2021.08.25-9d8d617"    # current set of TOAs available
