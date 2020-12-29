import numpy as np

class DMXParameter:
    """
    Convenience class for DMX parameters.
    """
    aliases = {'idx':'index', 'val':'dmx_val', 'err':'dmx_err', 'ep':'epoch',
               'r1':'low_mjd', 'r2':'high_mjd', 'f1':'low_freq',
               'f2':'high_freq', 'mask':'toa_mask'}
    def __init__(self):
        """
        """
        self.idx = 0  # index label [int]
        self.val = 0.0  # DMX value [cm**-3 pc]
        self.err = 0.0  # DMX uncertainty [cm**-3 pc]
        self.ep = 0.0  # epoch [MJD]
        self.r1 = 0.0 # left bin edge [MJD]
        self.r2 = 0.0 # right bin edge [MJD]
        self.f1 = 0.0 # lowest frequency [MHz]
        self.f2 = 0.0 # highest frequency [MHz]
        self.mask = []  # Boolean index array for selecting TOAs

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'aliases':
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)

    def print_dmx(self, range_only=False, fit_flag=True, fortran=False):
        """
        Print TEMPO-style DMX parameter.

        range_only=True will print only the bin edges [MJD].
        fit_flag=False will set the fit flag to 0 instead of 1.
        fortran=True will use 'D' instead of 'e'.
        """
        if range_only:
            DMX_str = (
                      f'DMXR1_{self.idx:04d}{self.r1:16.5f}\n'
                      f'DMXR2_{self.idx:04d}{self.r2:16.5f}\n'
                      )
            print(DMX_str)
        else:
            fit_flag = int(fit_flag)
            DMX_str = (
                      f'DMX_{self.idx:04d}{self.val:18.8e}{fit_flag:3}{self.err:20.8e}\n'
                      f'DMXEP_{self.idx:04d}{self.ep:16.5f}\n'
                      f'DMXR1_{self.idx:04d}{self.r1:16.5f}\n'
                      f'DMXR2_{self.idx:04d}{self.r2:16.5f}\n'
                      f'DMXF1_{self.idx:04d}{self.f1:16.3f}\n'
                      f'DMXF2_{self.idx:04d}{self.f2:16.3f}\n'
                      )
            if fortran: DMX_str = DMX_str.replace('e','D')
            print(DMX_str)


def get_dmx_ranges(toas, bin_width=1.0, pad=0.0, strict_inclusion=True):
    """
    Returns a list of low and high MJDs defining DMX ranges, covering all TOAs.

    NB: all returned items are sorted according to MJD.

    This emulates the TEMPO binning algorithm in $TEMPO/src/arrtim.f, but not
        exactly.

    toas is a PINT TOA object
    bin_width is the largest permissible DMX bin width [d]
    pad is the additional amount of time [d] added to the beginning and end of
        every DMX range.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    # Get TOA info
    mjds = toas.get_mjds().value  # day
    isort = mjds.argsort()  # just to be safe
    mjds = mjds[isort]      # order them

    # Initialize lists to be returned
    DMXR1, DMXR2 = [], []  # low and high MJDs (bin edges), respectively

    # Now step through the TOAs, starting with the first (earliest)
    remaining_mjds = np.copy(mjds)
    while(len(remaining_mjds)):
        left_bin_edge = remaining_mjds.min() - pad
        if strict_inclusion: left_bin_edge -= 1e-11  # ~1 us
        right_bin_edge = left_bin_edge + bin_width + pad
        if strict_inclusion: right_bin_edge += 1e-11  # ~1 us
        in_bin = (left_bin_edge <= remaining_mjds) & \
                (remaining_mjds <= right_bin_edge)
        not_in_bin = np.logical_not(in_bin)  # currently not used
        right_bin_edge = remaining_mjds[in_bin].max() + pad # shrink bin
        if strict_inclusion: right_bin_edge += 1e-11  # ~1 us
        if np.any(in_bin):  # this should always be True
            DMXR1.append(left_bin_edge)
            DMXR2.append(right_bin_edge)
            not_accounted_for = remaining_mjds > right_bin_edge
        else:
            print("No TOAs in proposed DMX bin. Something is wrong.")
        # Update remaining TOA MJDs
        remaining_mjds = remaining_mjds[not_accounted_for]

    # Check that all are in a bin, and only one bin
    DMX_masks = [(left_bin_edge < mjds) & (mjds < right_bin_edge) for \
            left_bin_edge,right_bin_edge in zip(DMXR1,DMXR2)]
    assert sum([sum(DMX_mask) for DMX_mask in DMX_masks]) == len(mjds)

    dmx_ranges = list(zip(DMXR1,DMXR2))

    return dmx_ranges


def check_coverage(toas, ranges):
    """
    """


def get_dmx_mask(toas, low_mjd, high_mjd, sort=False, strict_inclusion=True):
    """
    Return a Boolean index array for selecting TOAs from toas.

    toas is a PINT TOA object of toas in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    sort=True returns the mask as though the TOAs are already sorted by MJD.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = toas.get_mjds().value  # day
    if sort:
        isort = mjds.argsort()  # indices to sort by
        mjds = mjds[isort]  # order them

    if strict_inclusion:
        mask = (low_mjd < mjds) & (mjds < high_mjd)
    else:  # NB: the following could allow for TOAs in multiple bins
        mask = (low_mjd <= mjds) & (mjds <= high_mjd)

    return mask


def get_dmx_epoch(toas, weighted_average=True):
    """
    Return the epoch of a DMX bin.

    toas is a PINT TOA object of toas in the DMX bin.
    weighted_average=True uses the TOA uncertainties as weights in determining
        the DMX epoch.
    """

    mjds = toas.get_mjds().value  # day

    if weighted_average:
        errs = toas.get_errors().value  # us
        weights = errs**-2
    else:
        weights = np.ones(len(toas))

    epoch = np.average(mjds, weights=weights)

    return epoch


def get_dmx_freqs(toas, allow_wideband=True):
    """
    Return the lowest and highest frequency of the TOAs in a DMX bin.

    toas is a PINT TOA object of toas in the DMX bin.
    allow_wideband=True will consider the -fratio and -bw flags in the
        determination of these frequencies, if toas contains wideband TOAs.
    """

    freqs = toas.get_freqs().value  # MHz
    high_freq = 0.0
    low_freq = np.inf

    # indices of wideband TOAs
    iwb = np.arange(len(toas))[np.array(toas.get_flag_value('pp_dm')[0]) \
            != None]
    if allow_wideband:  # the following arrays will be empty if narrowband TOAs
        fratios = toas[iwb].get_flag_value('fratio') # frequency ratio / WB TOA
        fratios = np.array(fratios[0])
        bws = toas[iwb].get_flag_value('bw')  # bandwidth [MHz] / WB TOA
        bws = np.array(bws[0])
        low_freqs = bws / (fratios - 1)
        high_freqs = bws + low_freqs

    for itoa in range(len(toas)):
        if itoa in iwb and allow_wideband:
            if low_freqs[itoa] < low_freq: low_freq = low_freqs[itoa]
            if high_freqs[itoa] > high_freq: high_freq = high_freqs[itoa]
        else:
            if freqs[itoa] < low_freq: low_freq = freqs[itoa]
            if freqs[itoa] > high_freq: high_freq = freqs[itoa]

    return low_freq, high_freq


def make_dmx(toas, dmx_ranges, dmx_vals=None, dmx_errs=None, sort=False,
        strict_inclusion=True, weighted_average=True, allow_wideband=True,
        start_idx=1, print_dmx=False):
    """
    Uses convenience functions to assemble a TEMPO-style DMX parameters.

    toas is a PINT TOA object of toas in the DMX bin.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    dmx_vals is an array of DMX parameter values [pc cm**-3]; defaults to
        zeros.
    dmx_errs is an array of DMX parameter uncertainties [pc cm**-3]; defaults
        to zeros.
    sort and strict_inclusion are kwargs passed to get_dmx_mask().
    weighted_average is a kwarg passed to get_dmx_epoch().
    allow_wideband is a kwarg passed to get_dmx_freqs().
    start_idx is the index label of the first DMX parameter, which will
        increment with each entry in dmx_ranges; there is no assumption about
        MJD ordering.
    print_dmx=True will print all DMX parameters in TEMPO style.
    """

    dmx_parameters = []

    if dmx_vals is None:
        dmx_vals = np.zeros(len(toas))
        dmx_errs = np.zeros(len(toas))

    for idmx,idx in enumerate(range(start_idx, start_idx+len(dmx_ranges))):
        low_mjd = min(dmx_ranges[idmx])
        high_mjd = max(dmx_ranges[idmx])
        mask = get_dmx_mask(toas, low_mjd, high_mjd, sort, strict_inclusion)
        epoch = get_dmx_epoch(toas[mask], weighted_average)
        low_freq, high_freq = get_dmx_freqs(toas, allow_wideband)
        dmx_parameter = DMXParameter()
        dmx_parameter.idx = idx
        dmx_parameter.val = dmx_vals[idmx]
        dmx_parameter.err = dmx_vals[idmx]
        dmx_parameter.ep = epoch
        dmx_parameter.r1 = low_mjd
        dmx_parameter.r2 = high_mjd
        dmx_parameter.f1 = low_freq
        dmx_parameter.f2 = high_freq
        dmx_parameter.mask = mask
        dmx_parameters.append(dmx_parameter)
        if print_dmx: dmx_parameter.print_dmx(range_only=False, fit_flag=True,
                fortran=False)

    return dmx_parameters


def add_dmx_range():
    """
    Will call PINT model method function.
    """
    pass

def remove_dmx_range():
    """
    Will call PINT model method function.
    """
    pass
