import numpy as np
from astropy import log
from timing_analysis.utils import apply_cut_flag, apply_cut_select

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
    for mjd in sorted(mjds):  # loop over sorted MJDs
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


def get_dmx_ranges(toas, bin_width=1.0, pad=0.0, strict_inclusion=True,
        check=True):
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
    check=True executes check_dmx_ranges() on the TOAs and the output DMX
        ranges.
    """

    # Order the MJDs
    mjds = toas.get_mjds().value
    isort = mjds.argsort()
    mjds = mjds[isort]

    # Initialize lists to be returned
    low_mjds, high_mjds = [], []  # low and high MJDs (bin edges), respectively

    # Now step through the MJDs, starting with the first (earliest)
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
            msg = "No TOAs in proposed DMX bin. Something is wrong."
            log.warning(msg)
        # Update remaining TOA MJDs
        remaining_mjds = remaining_mjds[not_accounted_for]

    dmx_ranges = list(zip(low_mjds,high_mjds))  # Should already be sorted

    # Check that all TOAs are in a bin and only one bin, and no empty ranges
    if check: check_dmx_ranges(toas, dmx_ranges)

    return dmx_ranges


def get_gasp_dmx_ranges(toas, group_width=0.1, bin_width=15.0, pad=0.0,
        strict_inclusion=True, check=True):
    """
    Return a list of DMX ranges that group GASP TOAs into bins.

    NB: Adopted from dmx_fixer.py.

    toas is a PINT TOA object.
    group_width [day] is a kwarg passed to group_dates().
    bin_width is the largest permissible DMX bin width [d].
    pad is the additional amount of time [d] added to the beginning and end of
        every DMX range.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    check=True executes check_dmx_ranges() on the TOAs and the output DMX
        ranges.
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

    # Initialize lists to be returned
    low_mjds, high_mjds = [], []  # low and high MJDs (bin edges), respectively

    # Pair up high and low frequency GASP observations
    febes = np.array(toas.get_flag_value('f')[0])  # frontend_backend flag
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
                strict_inclusion=strict_inclusion, add_new_ranges=False,
                check=False)

    # Check that GASP TOAs are in a bin and only one bin, and no empty ranges
    if check: check_dmx_ranges(toas, dmx_ranges)

    return dmx_ranges


def expand_dmx_ranges(toas, dmx_ranges, bin_width=1.0, pad=0.0,
        strict_inclusion=True, add_new_ranges=False, check=True):
    """
    Expands DMX ranges to accommodate new TOAs up to a maximum bin width.

    Returns a list of DMX ranges.

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
    check=True executes check_dmx_ranges() on the TOAs and the new DMX ranges.
    """

    dmx_ranges = sorted(dmx_ranges, key=lambda dmx_range: dmx_range[0])

    if not len(dmx_ranges):  # in case an empty list was passed
        if add_new_ranges:
            dmx_ranges += get_dmx_ranges(toas, bin_width=bin_width, pad=pad,
                    strict_inclusion=strict_inclusion, check=check)
        return dmx_ranges

    # Get the TOAs that don't have a DMX bin (inone)
    masks, ibad, iover, iempty, inone, imult = check_dmx_ranges(toas,
            dmx_ranges, full_return=True, quiet=True)

    iremain = []  # for the TOAs that are not accommodated by expansion

    dmx_ranges = np.array(dmx_ranges)  # tuples don't support item assignment

    for itoa in inone:
        expanded = False
        mjd = toas.get_mjds().value[itoa]
        inearest = np.argmin(abs(mjd - dmx_ranges.flatten()))
        irange = int(inearest / 2)  # index of the nearest dmx range
        left_bin_edge = dmx_ranges[irange][0]
        right_bin_edge = dmx_ranges[irange][1]
        if (left_bin_edge < mjd) & (mjd < right_bin_edge):  # in a bin
            continue  # previous iteration of loop could now cover this TOA
        width = right_bin_edge - left_bin_edge
        if not (inearest % 2):  # TOA is outside left bin-edge, has smaller MJD
            delta_mjd = left_bin_edge - mjd  # must be positive
            if width + delta_mjd < bin_width:  # can the range be widened?
                dmx_ranges[irange][0] -= delta_mjd + pad
                if strict_inclusion: dmx_ranges[irange][0] -= 1e-11  # ~1 us
                expanded = True
        else:  # TOA is outside right bin-edge, has larger MJD
            delta_mjd = mjd - right_bin_edge  # must be positive
            if width + delta_mjd < bin_width:  # can the range be widened?
                dmx_ranges[irange][1] += delta_mjd + pad
                if strict_inclusion: dmx_ranges[irange][1] += 1e-11  # ~1 us
                expanded = True
        if not expanded:  # TOA still not included
            iremain.append(itoa)

    dmx_ranges = list(map(tuple, dmx_ranges)) # return to list of tuples

    if add_new_ranges:
        dmx_ranges += get_dmx_ranges(toas[iremain], bin_width=bin_width,
                pad=pad, strict_inclusion=strict_inclusion, check=False)
    dmx_ranges = sorted(dmx_ranges, key=lambda dmx_range: dmx_range[0])

    # Check that all TOAs are in a bin and only one bin, and no empty ranges
    if check: check_dmx_ranges(toas, dmx_ranges)

    return dmx_ranges


def check_dmx_ranges(toas, dmx_ranges, full_return=False, quiet=False):
    """
    Ensures all TOAs match only one DMX bin and all bins have at least one TOA.

    Also checks for improperly set and overlapping bins.

    NB: range boundaries are exclusive (strict_inclusion).

    Returns (if full_return)--
        masks: TOA masks for each range.
        ibad: Indices of bad/improper ranges.
        iover: Indices of ranges that overlap with one another.
        iempty: Indices of ranges with no TOAs assigned.
        inone: Indices of TOAs not assigned to any range.
        imult: Indices of TOAs assigned to multiple ranges.

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

    ibad, iover, iempty, inone, imult = [], [], [], [], []

    # Check for bad bins
    for irange,dmx_range in enumerate(dmx_ranges):
        left_bin_edge, right_bin_edge = dmx_range[0], dmx_range[1]
        if left_bin_edge == right_bin_edge or left_bin_edge > right_bin_edge:
            ibad.append(irange)
            if not quiet:
                msg = f"DMX range with pythonic index {irange}, correponding to the DMX range {dmx_ranges[irange]}, is improper."
                log.info(msg)
    if len(ibad) and not quiet:
        msg = f"{len(ibad)} DMX ranges are improper."
        log.warning(msg)

    # Check for overlapping bins
    for irange,dmx_range in enumerate(dmx_ranges):
        left_bin_edge, right_bin_edge = dmx_range[0], dmx_range[1]
        other_ranges = \
                np.array(dmx_ranges)[np.where(np.arange(len(dmx_ranges)) \
                != irange)[0]]
        for mjd in other_ranges.flatten():
            if (left_bin_edge < mjd) & (mjd < right_bin_edge):  # finds overlap
                iover.append(irange)
                if not quiet:
                    msg = f"DMX range with pythonic index {irange}, correponding to the DMX range {dmx_ranges[irange]}, overlaps with other ranges."
                    log.info(msg)
                break  # only find first overlap
    if len(iover) and not quiet:
        msg = f"{len(iover)} DMX ranges are overlapping with one another."
        log.warning(msg)

    # Check for empty bins
    ntoa_per_dmx_bin = np.array(masks).astype(int).sum(axis=1)
    if not np.all(ntoa_per_dmx_bin):
        iempty = np.where(ntoa_per_dmx_bin == 0)[0]
        for irange in iempty:
            if not quiet:
                msg = f"DMX range with pythonic index {irange}, correponding to the DMX range {dmx_ranges[irange]}, overlaps no TOAs."
                log.info(msg)
        if not quiet:
            msg = f"{len(iempty)} DMX ranges have no TOAs."
            log.warning(msg)

    # Check each TOA in exactly one bin
    ndmx_per_toa = np.array(masks).astype(int).sum(axis=0)
    if not np.all(ndmx_per_toa == 1):
        inone = np.where(ndmx_per_toa < 1)[0]  # not one bin
        for itoa in inone:
            if not quiet:
                msg = f"TOA with index {itoa} (MJD {mjds[itoa]}, {toas.get_freqs().value[itoa]} MHz) does not have a DMX range."
                log.info(msg)
        imult = np.where(ndmx_per_toa > 1)[0]  # multiple bins
        for itoa in imult:
            irange = list(np.where(np.array(masks).astype(int)[:,itoa] \
                    == 1)[0])
            if not quiet:
                msg = f"TOA with index {itoa} (MJD {mjds[itoa]}, {toas.get_freqs().value[itoa]} MHz) is in {ndmx_per_toa[itoa]} DMX ranges (with pythonic indices {irange})."
                log.info(msg)
        if not quiet:
            msg = f"{len(inone)} TOAs have no DMX range and {len(imult)} TOAs are in multiple DMX ranges."
            log.warning(msg)

    if full_return:
        return masks, ibad, iover, iempty, inone, imult


def get_dmx_mask(toas, low_mjd, high_mjd, strict_inclusion=True):
    """
    Return a Boolean index array for selecting TOAs from toas in a DMX range.

    toas is a PINT TOA object of TOAs in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = toas.get_mjds().value

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
        print(fratios)
        low_freqs = bws.astype('float32') / (fratios.astype('float32') - 1)
        high_freqs = bws.astype('float32') + low_freqs

    for itoa in range(len(toas)):
        if itoa in iwb and allow_wideband:
            if low_freqs[itoa] < low_freq: low_freq = low_freqs[itoa]
            if high_freqs[itoa] > high_freq: high_freq = high_freqs[itoa]
        else:
            if freqs[itoa] < low_freq: low_freq = freqs[itoa]
            if freqs[itoa] > high_freq: high_freq = freqs[itoa]

    return low_freq, high_freq


def check_frequency_ratio(toas, dmx_ranges, frequency_ratio=1.1,
        strict_inclusion=True, allow_wideband=True, invert=False, quiet=False):
    """
    Check that the TOAs in a DMX bin pass a frequency ratio criterion.

    Returns the indices of the TOAs and DMX ranges that pass the test.

    toas is a PINT TOA object.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    frequency_ratio is the ratio of high-to-low frequencies in the DMX bin;
        the frequencies used are returned by get_dmx_freqs().
    strict_inclusion is a Boolean kwarg passed to get_dmx_mask().
    allow_wideband is a Boolean kwarg passed to get_dmx_freqs(); if True, the
        bandwidths of the wideband TOAs are considered in the calculation.
    invert=True will return the indices of the TOAs and DMX ranges that fail to
        pass the test.
    quiet=True turns off the logged info.
    """

    toa_mask = np.zeros(len(toas), dtype=bool)  # selects TOAs and ranges
    dmx_range_mask = np.zeros(len(dmx_ranges), dtype=bool)  # that pass

    nfail_toas = 0
    for irange,dmx_range in enumerate(dmx_ranges):
        low_mjd, high_mjd = dmx_range[0], dmx_range[1]
        mask = get_dmx_mask(toas, low_mjd, high_mjd,
                strict_inclusion=strict_inclusion)
        low_freq, high_freq = get_dmx_freqs(toas[mask],
            allow_wideband=allow_wideband)
        if high_freq / low_freq >= frequency_ratio:  # passes
            toa_mask += mask
            dmx_range_mask[irange] = True
        else:  # fails
            nfail_toas += len(toas[mask])
            if not quiet:
                msg = f"DMX range with pythonic index {irange}, correponding to the DMX range {dmx_ranges[irange]}, contains TOAs that do not pass the frequency ratio test (TOAs with MJDs {toas[mask].get_mjds().value})."
                log.info(msg)

    nfail_ranges = sum(np.logical_not(dmx_range_mask))
    if not quiet and nfail_ranges:
        msg = f"{nfail_ranges} DMX ranges, which include {nfail_toas} TOAs, do not pass the frequency ratio test."
        log.warning(msg)

    if not invert:  #  return those that pass
        return np.arange(len(toas))[toa_mask], \
                np.arange(len(dmx_ranges))[dmx_range_mask]
    else:  # return those that fail
        return np.arange(len(toas))[np.logical_not(toa_mask)], \
                np.arange(len(dmx_ranges))[np.logical_not(dmx_range_mask)]


def check_solar_wind(toas, dmx_ranges, model, max_delta_t=0.1, bin_width=1.0,
        solar_n0=5.0, allow_wideband=True, strict_inclusion=True, pad=0.0,
        check=True, return_only=False, quiet=False):
    """
    Split DMX ranges based on influence of the solar wind.

    Returns a list of DMX ranges.

    Uses approximation for SW model until PINT's model is improved/working.

    NB: Adopted from dmx_fixer.py.

    toas is a PINT TOA object.
    model is a PINT model object, which is currently how one accesses the solar
        elogation angle of the TOAs.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    max_delta_t is the time delay [us] above which a DMX range will be split.
    bin_width is the largest permissible DMX bin width [d] for the new ranges.
    solar_n0 is the solar wind electron desity [cm**-3] at 1 AU to use in the
        model.
    allow_wideband is a Boolean kwarg passed to get_dmx_freqs(); if True, the
        bandwidths of the wideband TOAs are considered in the calculation.
    strict_inclusion is a Boolean kwarg passed to get_dmx_mask and
        get_dmx_ranges().
    pad is a float kwarg passed to get_dmx_ranges().
    check=True executes check_dmx_ranges() on the TOAs and the output DMX
        ranges.
    return_only=True will return only the indices of problematic ranges and
        TOAs.
    quiet=True turns off the logged info.
    """
    # constants in the model
    #one_AU = 499.005  # 1 astronomical unit [s]
    one_AU = 4.8481e-6  # 1 astronomical unit [pc]
    Dconst = 2.41e-4  # "inverse" dispersion constant [MHz**-2 pc cm**-3 s**-1]

    # Get the solar elongation angle [rad]
    phis = model.sun_angle(toas, heliocenter=True, also_distance=False).value

    toa_mask = np.zeros(len(toas), dtype=bool)  # selects problem TOAs
    dmx_range_mask = np.zeros(len(dmx_ranges), dtype=bool)  # and ranges

    for irange,dmx_range in enumerate(dmx_ranges):
        low_mjd, high_mjd = dmx_range[0], dmx_range[1]
        mask = get_dmx_mask(toas, low_mjd, high_mjd,
                strict_inclusion=strict_inclusion)
        low_freq, high_freq = get_dmx_freqs(toas[mask],
            allow_wideband=allow_wideband)
        # Convert to time delay, using calc from David's code (fixed)
        theta = np.pi - phis[mask]  # rad
        #Excess DM from Solar wind (approximate)
        #delta_dm = theta * (solar_n0 / 10.) * 2.4098e-2 / \
        #        (one_AU * abs(np.sin(theta)))  # pc cm**-3
        delta_dm = solar_n0 * one_AU * theta / abs(np.sin(theta))
        dm_delays = delta_dm / (Dconst * low_freq**2) * 1e6  # us
        delta_t = max(dm_delays) - min(dm_delays)
        if delta_t > max_delta_t:
            toa_mask += mask
            dmx_range_mask[irange] = True
            if not quiet:
                msg = f"DMX range with pythonic index {irange}, correponding to the DMX range {dmx_ranges[irange]}, contains TOAs that are affected by the solar wind (TOAs with MJDs {toas[mask].get_mjds().value})."
                log.info(msg)
    nsolar = sum(dmx_range_mask)
    if not quiet and nsolar:
        msg = f"{nsolar} DMX ranges are affected by the solar wind."
        log.warning(msg)

    if return_only:  # return indices of problem TOAs and ranges
        return np.arange(len(toas))[toa_mask], \
                np.arange(len(dmx_ranges))[dmx_range_mask]
    else:  # return augmented ranges
        if nsolar:  # if there
            # Select good ranges
            dmx_ranges = np.array(dmx_ranges)[np.logical_not(dmx_range_mask)]
            dmx_ranges = list(map(tuple, dmx_ranges)) # return to tuples
            dmx_ranges += get_dmx_ranges(toas[toa_mask], bin_width=bin_width,
                    pad=pad, strict_inclusion=strict_inclusion)  # add new ones
            dmx_ranges = sorted(dmx_ranges, key=lambda dmx_range: dmx_range[0])

            # Check that all TOAs are in a bin and only one bin, no empty range
            if check: check_dmx_ranges(toas, dmx_ranges)

        return dmx_ranges


def add_dmx(model, bin_width=1.0):
    """
    Checks for DispersionDMX and ensures the bin width is the only parameter.

    model is a PINT model object.
    bin_width is the largest permissible DMX bin width [d].
    """

    if 'DispersionDMX' not in model.components.keys():
        from pint.models.timing_model import Component
        dmx_component = Component.component_types["DispersionDMX"]
        model.add_component(dmx_component())
        dmx = model.components["DispersionDMX"]
        dmx.remove_DMX_range(1)  # remove the range it automatically initiates
        dmx.DMX.set(bin_width)
    else:
        dmx = model.components["DispersionDMX"]
        dmx.DMX.set(bin_width)


def model_dmx_params(model):
    """
    Get DMX ranges, values, and uncertainties from a PINT model object.

    model is a PINT model object.
    """

    dmx_ranges, dmx_vals, dmx_errs = [], [], []

    if 'DispersionDMX' in model.components.keys():
        dmx = model.components['DispersionDMX']
        idxs = dmx.get_indices()
        for idx in idxs:
            low_mjd = getattr(model, f"DMXR1_{idx:04d}").value
            high_mjd = getattr(model, f"DMXR2_{idx:04d}").value
            dmx_val = getattr(model, f"DMX_{idx:04d}").value
            try: dmx_err = getattr(model, f"DMX_{idx:04d}").uncertainty.value
            except AttributeError: dmx_err = 0.0
            dmx_ranges.append((low_mjd, high_mjd))
            dmx_vals.append(dmx_val)
            dmx_errs.append(dmx_err)
        dmx_vals = np.array(dmx_vals)
        dmx_errs = np.array(dmx_errs)

    return dmx_ranges, dmx_vals, dmx_errs


def remove_all_dmx_ranges(model, quiet=False):
    """
    Uses PINT to remove all DMX parameter ranges from a timing model.

    model is a PINT model object.
    quiet=True turns off the logged info.
    """

    if 'DispersionDMX' in model.components.keys():
        dmx = model.components['DispersionDMX']
        idxs = dmx.get_indices()
        for idx in idxs:
            dmx.remove_DMX_range(idx)  #remove parameters
        if not quiet:
            msg = f"Removed {len(idxs)} DMX parameters from timing model."
            log.info(msg)
    else:
        pass


def setup_dmx(model, toas, quiet=True, frequency_ratio=1.1, max_delta_t=0.1,
        freeze_DM=True):
    """
    Sets up and checks a DMX model using a number of defaults.

    Returns new PINT TOA object containing TOAs passing frequency ratio test.

    model is a PINT model object.
    toas is a PINT TOA object.
    frequency_ratio is the ratio of high-to-low frequencies in the DMX bin;
        the frequencies used are returned by get_dmx_freqs().
    max_delta_t is the time delay [us] above which a DMX range will be split.
    quiet=True turns off some of the logged warnings and info.
    freeze_DM=True ensures the mean DM parameter is not fit.
    """

    # Freeze DM
    model.DM.frozen = freeze_DM

    # Get existing DMX ranges and values from model; adjust the mean DM
    old_dmx_ranges, old_dmx_vals, old_dmx_errs = model_dmx_params(model)
    if len(old_dmx_ranges) and np.sum(old_dmx_errs):
        idmx = old_dmx_errs != 0.0
        mean_old_dmx_val, sum_of_weights = np.average(old_dmx_vals[idmx],
                weights=old_dmx_errs[idmx]**-2, returned=True)
        mean_old_dmx_val_err = sum_of_weights**-0.5
        if abs(mean_old_dmx_val / mean_old_dmx_val_err) > 1.0:
            adjust_old_dmx = True
            old_DM = model.DM.value
            DM = model.DM.value + mean_old_dmx_val
            model.DM.set(DM)
            msg = f"Updated mean DM parameter from {old_DM:.7f} to {DM:.7f}."
            log.info(msg)
        else:
            adjust_old_dmx = False
    else:
        adjust_old_dmx = False

    # Set up DMX model
    if toas.observatories == set(['arecibo']): bin_width = 0.5  # day
    else: bin_width = 6.5  #day
    # Calculate GASP-era ranges, if applicable
    dmx_ranges = get_gasp_dmx_ranges(toas, group_width=0.1, bin_width=15.0,
            pad=0.05, check=False)
    # Now expand to include all TOAs
    dmx_ranges = expand_dmx_ranges(toas, dmx_ranges, bin_width=bin_width,
            pad=0.05, add_new_ranges=True, check=False)

    # Ensure DM events have fine DMX binning --> need to generalize later
    if model.PSR.value == 'J1713+0747':
        toa_mask = np.zeros(len(toas), dtype=bool)
        dmx_range_mask = np.ones(len(dmx_ranges), dtype=bool)
        for irange,dmx_range in enumerate(dmx_ranges):
            low_mjd, high_mjd = dmx_range[0], dmx_range[1]
            # if it overlaps with the event at all...
            if ((57508 < low_mjd) and (low_mjd < 57513)) or \
                    ((57508 < high_mjd) and (high_mjd < 57513)):
                        dmx_range_mask[irange] = False
                        toa_mask += get_dmx_mask(toas, low_mjd, high_mjd)
        # Select the non-event ranges
        dmx_ranges = np.array(dmx_ranges)[dmx_range_mask]
        dmx_ranges = list(map(tuple, dmx_ranges))
        # Add new ranges based on the TOAs during the event
        dmx_ranges += get_dmx_ranges(toas[toa_mask], bin_width=0.5,
                check=False)
        # Sort the ranges
        dmx_ranges = sorted(dmx_ranges, key=lambda dmx_range: dmx_range[0])

    # Do basic checks of DMX model
    dmx_ranges = check_solar_wind(toas, dmx_ranges, model,
            max_delta_t=max_delta_t, bin_width=0.5, pad=0.05, check=False,
            quiet=quiet)
    itoas, iranges = check_frequency_ratio(toas, dmx_ranges,
            frequency_ratio=frequency_ratio, quiet=quiet)

    # Find TOAs failing fratio test and apply cuts
    ftoas, franges = check_frequency_ratio(toas, dmx_ranges,
            frequency_ratio=frequency_ratio, quiet=quiet, invert=True)
    fratio_inds = toas.table['index'][ftoas]
    if len(fratio_inds):
        apply_cut_flag(toas,fratio_inds,'dmx')
        apply_cut_select(toas,reason='frequency ratio check')

    dmx_ranges = np.array(dmx_ranges)[iranges]
    dmx_ranges = list(map(tuple, dmx_ranges))

    # Sort the ranges
    dmx_ranges = sorted(dmx_ranges, key=lambda dmx_range: dmx_range[0])

    # Check for sanity and use inone to flag dmx cuts
    masks, ibad, iover, iempty, inone, imult = \
            check_dmx_ranges(toas, dmx_ranges, full_return=True, quiet=False)

    if len(ibad) + len(iover) + len(iempty) + len(inone) + len(imult) == 0:
        msg = "Proposed DMX model OK."
        log.info(msg)
    else:
        pass  # check_dmx_ranges will print lots of warnings if quiet=False

    if len(dmx_ranges) == len(old_dmx_ranges):
        old_dmx_ranges = sorted(old_dmx_ranges,  #  sort the old ranges
                key=lambda dmx_range: dmx_range[0])
        if np.isclose(np.array(dmx_ranges).flatten(), \
                np.array(old_dmx_ranges).flatten(), \
                rtol=1e-10, atol=1e-5).all():
            msg = f"Proposed DMX ranges are the same as input; keeping input DMX model."
            log.info(msg)
            if adjust_old_dmx:
                dmx = model.components['DispersionDMX']
                idxs = dmx.get_indices()
                for idx in idxs:
                    low_mjd = getattr(model, f"DMXR1_{idx:04d}").value
                    high_mjd = getattr(model, f"DMXR2_{idx:04d}").value
                    dmx_val = getattr(model, f"DMX_{idx:04d}").value
                    dmx_val -= mean_old_dmx_val
                    frozen = getattr(model, f"DMX_{idx:04d}").frozen
                    dmx.remove_DMX_range(idx)
                    dmx.add_DMX_range(low_mjd, high_mjd, idx, dmx_val,
                            frozen=frozen)
                    msg = f"Subtracted {mean_old_dmx_val:.7f} from existing DMX values."
                log.info(msg)
            return toas

    # Remove all DMX bins from model
    remove_all_dmx_ranges(model, quiet=False)

    # Add DMX parameters to model
    add_dmx(model, bin_width)
    dmx = model.components['DispersionDMX']
    for irange,dmx_range in enumerate(dmx_ranges):
        dmx.add_DMX_range(dmx_range[0], dmx_range[1], index=irange+1,
                dmx=0.0, frozen=False)
    msg = f"Added {len(dmx_ranges)} DMX parameters to timing model."
    log.info(msg)

    return toas


def make_dmx(toas, dmx_ranges, dmx_vals=None, dmx_errs=None,
        strict_inclusion=True, weighted_average=True, allow_wideband=True,
        start_idx=1, print_dmx=False):
    """
    Uses convenience functions to assemble a TEMPO-style DMX parameters.

    Returns list of DMXParameter objects.

    toas is a PINT TOA object.
    dmx_ranges is a list of (low_mjd, high_mjd) pairs defining the DMX ranges;
        see the output of get_dmx_ranges().
    dmx_vals is an array of DMX parameter values [pc cm**-3]; defaults to
        zeros.
    dmx_errs is an array of DMX parameter uncertainties [pc cm**-3]; defaults
        to zeros.
    strict_inclusion is a Boolean kwarg passed to get_dmx_mask().
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

    for irange,idx in enumerate(range(start_idx, start_idx+len(dmx_ranges))):
        low_mjd = min(dmx_ranges[irange])
        high_mjd = max(dmx_ranges[irange])
        mask = get_dmx_mask(toas, low_mjd, high_mjd, strict_inclusion)
        epoch = get_dmx_epoch(toas[mask], weighted_average)
        low_freq, high_freq = get_dmx_freqs(toas[mask], allow_wideband)
        dmx_parameter = DMXParameter()
        dmx_parameter.idx = idx
        dmx_parameter.val = dmx_vals[irange]
        dmx_parameter.err = dmx_vals[irange]
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
