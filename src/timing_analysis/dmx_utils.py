import numpy as np
from astropy import log

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
        self.ep = 0.0  # epoch [day]
        self.r1 = 0.0 # left bin edge [day]
        self.r2 = 0.0 # right bin edge [day]
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

        range_only=True will print only the bin edges [day].
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


def group_dates(toas, group_width=0.1):
    """
    Returns MJDs of groups of TOAs no wider than a specified amount.

    NB: originally from dmx_fixer.py.

    toas is a PINT TOA object.
    group_width specifies the maximum width of the TOA group [day]; it defines
        and 'epoch'.
    """

    mjds = toas.get_mjds().value

    group_mjds = [0.0]
    igroup = 0
    group_count = 0
    low_mjd = 0.0
    for mjd in sorted(mjds):
        if low_mjd > 0 and (mjd - low_mjd) > group_width:
            group_mjds[igroup] /= group_count
            igroup += 1
            group_count = 0.0
            group_mjds.append(0.0)
        group_mjds[igroup] += mjd
        group_count += 1
        low_mjd = mjd
    group_mjds[igroup] /= group_count

    group_mjds = np.array(group_mjds)

    return group_mjds


def get_dmx_ranges(toas, bin_width=1.0, pad=0.0, strict_inclusion=True):
    """
    Returns a list of low and high MJDs defining DMX ranges, covering all TOAs.

    NB: returned DMX ranges are sorted according to MJD.

    This emulates the TEMPO binning algorithm in $TEMPO/src/arrtim.f, but not
        exactly.

    toas is a PINT TOA object.
    bin_width is the largest permissible DMX bin width [d].
    pad is the additional amount of time [d] added to the beginning and end of
        every DMX range.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    # Get TOA info
    mjds = toas.get_mjds().value
    isort = mjds.argsort()  # just to be safe
    toas = toas[isort]      # order them
    mjds = mjds[isort]

    # Initialize lists to be returned
    low_mjds, high_mjds = [], []  # low and high MJDs (bin edges), respectively

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
            low_mjds.append(left_bin_edge)
            high_mjds.append(right_bin_edge)
            not_accounted_for = remaining_mjds > right_bin_edge
        else:
            log.warning("No TOAs in proposed DMX bin. Something is wrong.")
        # Update remaining TOA MJDs
        remaining_mjds = remaining_mjds[not_accounted_for]

    dmx_ranges = list(zip(low_mjds,high_mjds))

    # Check that all TOAs are in a bin and only one bin, and no empty ranges
    check_dmx_coverage(toas, dmx_ranges)

    return dmx_ranges


def get_gasp_dmx_ranges(toas, group_width=0.1, bin_width=15.0, pad=0.0,
        strict_inclusion=True):
    """
    Return a list of DMX ranges GASP TOAs into DMX ranges

    NB: Adopted from dmx_fixer.py.

    toas is a PINT TOA object.
    group_width [day] is a kwarg passed to group_dates().
    bin_width is the largest permissible DMX bin width [d].
    pad is the additional amount of time [d] added to the beginning and end of
        every DMX range.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    # Check that the TOAs contain GASP data
    bes = np.array(toas.get_flag_value('be')[0])  # backend flag
    if 'GASP' not in set(bes):
        return []

    # Only consider GASP data outside the GUPPI range (PUPPI came later)
    end_gasp_era = min(toas[bes == 'GUPPI'].get_mjds().value)
    mjds = toas.get_mjds().value
    toas = toas[(mjds < end_gasp_era) & (bes == 'GASP')]
    mjds = mjds[(mjds < end_gasp_era) & (bes == 'GASP')]
    # Only consider GASP
    isort = mjds.argsort()  # just to be safe
    toas = toas[isort]      # order them
    mjds = mjds[isort]
    febes = toas.get_flag_value('f')[0]  # frontend_backend flag
    febes = np.array(febes)[isort]

    # Initialize lists to be returned
    low_mjds, high_mjds = [], []  # low and high MJDs (bin edges), respectively

    # Pair up high and low frequency GASP observations
    low_obs_mjds = group_dates(toas[febes == 'Rcvr_800_GASP'], group_width)
    high_obs_mjds = group_dates(toas[febes == 'Rcvr1_2_GASP'], group_width)
    iclosest_high = [abs(mjd-high_obs_mjds).argmin() for mjd in low_obs_mjds]
    iclosest_low = [abs(mjd-low_obs_mjds).argmin() for mjd in high_obs_mjds]
    for ihigh,ilow in enumerate(iclosest_low):
        if ihigh == iclosest_high[ilow]:  # found a pair
            delta_mjd = abs(high_obs_mjds[ihigh] - low_obs_mjds[ilow])
            if delta_mjd < bin_width:  # if the pair is close enough
                left_bin_edge = min(high_obs_mjds[ihigh] - group_width,
                        low_obs_mjds[ilow] - group_width) - pad
                if strict_inclusion: left_bin_edge -= 1e-11  # ~1 us
                right_bin_edge = max(high_obs_mjds[ihigh] + group_width,
                        low_obs_mjds[ilow] + group_width) + pad
                if strict_inclusion: right_bin_edge += 1e-11  # ~1 us
                low_mjds.append(left_bin_edge)
                high_mjds.append(right_bin_edge)

    dmx_ranges = list(zip(low_mjds,high_mjds))

    # It's possible some TOAs are missed in the above
    masks = [(left_bin_edge < mjds) & (mjds < right_bin_edge) for \
            left_bin_edge,right_bin_edge in dmx_ranges]
    ndmx_per_toa = np.array(masks).astype(int).sum(axis=0)
    inone = np.where(ndmx_per_toa == 0)[0]
    if len(inone):  # see if we can expand a range to accommodate
        dmx_ranges = expand_dmx_ranges(toas[inone], dmx_ranges,
                bin_width=bin_width, pad=pad,
                strict_inclusion=strict_inclusion, add_new_ranges=False)

    # Check that all TOAs are in a bin and only one bin, and no empty ranges
    check_dmx_coverage(toas, dmx_ranges)

    return dmx_ranges


def expand_dmx_ranges(toas, dmx_ranges, bin_width=1.0, pad=0.0,
        strict_inclusion=True, add_new_ranges=False):
    """
    Expands DMX ranges to accommodate new TOAs up to a maximum bin width.

    NB: returned DMX ranges are sorted according to MJD.

    toas is a PINT TOA object.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    bin_width is the largest permissible DMX bin width [d].
    pad is the additional amount of time [d] added to the beginning and end of
        every DMX range.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    add_new_ranges=True will add new DMX ranges if the TOAs cannot be
        accomodated.
    """

    # Get TOA info
    mjds = toas.get_mjds().value
    isort = mjds.argsort()  # just to be safe
    toas = toas[isort]      # order them
    mjds = mjds[isort]
    dmx_ranges = sorted(dmx_ranges, key=lambda tup: tup[0])

    # Get the TOAs that don't have a DMX bin (inone)
    masks, iempty, inone, imult = check_dmx_coverage(toas, dmx_ranges,
            full_return=True, quiet=True)

    iremain = []  # for the TOAs that are not accommodated by expansion

    dmx_ranges = np.array(dmx_ranges)  # tuples don't support item assignment

    for itoa in inone:
        expanded = False
        mjd = mjds[itoa]
        inearest = np.argmin(abs(mjd - dmx_ranges.flatten()))
        idmx = int(inearest / 2)  # index of the dmx range
        left_bin_edge = dmx_ranges[idmx][0]
        right_bin_edge = dmx_ranges[idmx][1]
        if (left_bin_edge < mjd) & (mjd < right_bin_edge):  # in a bin
            continue  # previous iteration of loop could now cover this TOA
        width = right_bin_edge - left_bin_edge
        if not (inearest % 2):  # TOA is outside left bin-edge, smaller MJD
            delta_mjd = left_bin_edge - mjd  # must be positive
            if width + delta_mjd < bin_width:  # can the range be widened?
                dmx_ranges[idmx][0] -= delta_mjd + pad
                if strict_inclusion: dmx_ranges[idmx][0] -= 1e-11  # ~1 us
                expanded = True
        else:  # TOA is outside right bin-edge, larger MJD
            delta_mjd = mjd - right_bin_edge  # must be positive
            if width + delta_mjd < bin_width:  # can the range be widened?
                dmx_ranges[idmx][1] += delta_mjd + pad
                if strict_inclusion: dmx_ranges[idmx][1] += 1e-11  # ~1 us
                expanded = True
        if not expanded:  # TOA still not included
            iremain.append(itoa)

    dmx_ranges = list(map(tuple, dmx_ranges)) # return to list of tuples

    if add_new_ranges:
        dmx_ranges += get_dmx_ranges(toas[iremain], bin_width=bin_width,
                pad=pad, strict_inclusion=strict_inclusion)
    dmx_ranges = sorted(dmx_ranges, key=lambda tup: tup[0])

    return dmx_ranges


def check_dmx_coverage(toas, dmx_ranges, full_return=False, quiet=False):
    """
    Ensures all TOAs match only one DMX bin and all bins have at least one TOA.

    NB: range boundaries are exclusive (strict_inclusion).

    toas is a PINT TOA object.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    full_return=True will return the masks, indices of problematic ranges, and
        indices of problematic TOAs.
    quiet=True turns off the logged info.
    """

    mjds = toas.get_mjds().value

    masks = [(left_bin_edge < mjds) & (mjds < right_bin_edge) for \
            left_bin_edge,right_bin_edge in dmx_ranges]

    iempty, inone, imult = [], [] ,[]

    # Check for empty bins
    ntoa_per_dmx_bin = np.array(masks).astype(int).sum(axis=1)
    if not np.all(ntoa_per_dmx_bin):
        iempty = np.where(ntoa_per_dmx_bin == 0)[0]
        for imask in iempty:
            if not quiet: log.info(f"DMX range with pythonic index {imask}, correponding to the DMX range {dmx_ranges[imask]}, overlaps no TOAs.")
        if not quiet: log.warning(f"{len(iempty)} DMX ranges have no TOAs.")

    # Check each TOA in exactly one bin
    ndmx_per_toa = np.array(masks).astype(int).sum(axis=0)
    if not np.all(ndmx_per_toa == 1):
        inone = np.where(ndmx_per_toa < 1)[0]  # not one bin
        for itoa in inone:
            if not quiet: log.info(f"TOA with index {itoa} (MJD {mjds[itoa]}, {toas.get_freqs().value[itoa]} MHz) does not have a DMX range.")
        imult = np.where(ndmx_per_toa > 1)[0]  # multiple bins
        for itoa in imult:
            imask = list(np.where(np.array(masks).astype(int)[:,itoa] == 1)[0])
            if not quiet: log.info("TOA with index {itoa} (MJD {mjds[itoa]}, {toas.get_freqs().value[itoa]} MHz) is in {ndmx_per_toa[itoa]} DMX ranges (with pythonic indices {imask}).")
        if not quiet: log.warning(f"{len(inone)} TOAs have no DMX range and {len(imult)} TOAs are in multiple DMX ranges.")

    if full_return:
        return masks, iempty, inone, imult


def get_dmx_mask(toas, low_mjd, high_mjd, sort=False, strict_inclusion=True):
    """
    Return a Boolean index array for selecting TOAs from toas.

    toas is a PINT TOA object of TOAs in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    sort=True returns the mask as though the TOAs are already sorted by MJD.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = toas.get_mjds().value
    if sort:
        isort = mjds.argsort()  # indices to sort by
        toas = toas[isort]      # order them
        mjds = mjds[isort]

    if strict_inclusion:
        mask = (low_mjd < mjds) & (mjds < high_mjd)
    else:  # NB: the following could allow for TOAs in multiple bins
        mask = (low_mjd <= mjds) & (mjds <= high_mjd)

    return mask


def get_dmx_epoch(toas, weighted_average=True):
    """
    Return the epoch of a DMX bin.

    toas is a PINT TOA object of TOAs in the DMX bin.
    weighted_average=True uses the TOA uncertainties as weights in determining
        the DMX epoch.
    """

    mjds = toas.get_mjds().value

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

    toas is a PINT TOA object of TOAs in the DMX bin.
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

    toas is a PINT TOA object.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    dmx_vals is an array of DMX parameter values [pc cm**-3]; defaults to
        zeros.
    dmx_errs is an array of DMX parameter uncertainties [pc cm**-3]; defaults
        to zeros.
    sort and strict_inclusion are Boolean kwargs passed to get_dmx_mask().
    weighted_average is a Boolean kwarg passed to get_dmx_epoch().
    allow_wideband is a Boolean kwarg passed to get_dmx_freqs().
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
        low_freq, high_freq = get_dmx_freqs(toas[mask], allow_wideband)
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
