"""
This code contains the TimingConfiguration class, which is used to load configuration files and perform
actions, with items then passed to the notebooks.

Very basic usage:
    from timingconfiguration import TimingConfiguration
    tc = TimingConfiguration(CONFIGFILE)
"""
import io
import os
import glob
import pint.toa as toa
import pint.models as model
import pint.fitter
import numpy as np
import astropy.units as u
from astropy import log
import yaml
from timing_analysis.utils import write_if_changed, apply_cut_flag, apply_cut_select
from timing_analysis.defaults import *

class TimingConfiguration:
    """
    This class contains the functionality to read
    from a configuration file and send that information
    to the timing notebooks.
    """
    def __init__(self, filename="config.yaml", tim_directory=None, par_directory=None):
        """
        Initialization method.

        Normally config files are written to be run from the root of a
        git checkout on the NANOGrav notebook server. If you want to run
        them from somewhere else, you may need to override these directories
        when you construct the TimingConfiguration object; this will not
        change what is recorded in the config file.

        Parameters
        ==========
        filename (optional) : path to the configuration file
        tim_directory (optional) : override the tim directory specified in the config
        par_directory (optional) : override the par directory specified in the config
        """
        self.filename = filename
        with open(filename) as FILE:
            self.config = yaml.load(FILE, Loader=yaml.FullLoader)
        self.tim_directory = self.config['tim-directory'] if tim_directory is None else tim_directory
        self.par_directory = self.config['par-directory'] if par_directory is None else par_directory
        self.skip_check = self.config['skip-check'] if 'skip-check' in self.config.keys() else ''

    def get_source(self):
        """ Return the source name """
        return self.config['source']

    def get_compare_model(self):
        """ Return the timing model file to compare with """
        if "compare-model" in self.config.keys() and self.config['compare-model'] is not None:
            return os.path.join(self.par_directory, self.config['compare-model'])
        return None

    def get_free_params(self, fitter):
        """Return list of free parameters"""
        if self.config["free-dmx"]:
            return self.config['free-params'] + [p for p in fitter.model.params if p.startswith("DMX_")]
        else:
            return self.config['free-params']

    def get_model_and_toas(self,usepickle=True,print_all_ignores=False):
        """Return the PINT model and TOA objects"""
        par_path = os.path.join(self.par_directory,self.config["timing-model"])
        toas = self.config["toas"]

        # Individual tim file
        if isinstance(toas, str):
            toas = [toas]

        BIPM = self.get_bipm()
        EPHEM = self.get_ephem()
        m = model.get_model(par_path)

        if m.PSR.value != self.get_source():
            log.warning(f'{self.filename} source entry does not match par file value ({m.PSR.value}).')

        picklefilename = os.path.basename(self.filename) + ".pickle.gz"
        # Merge toa_objects (check this works for list of length 1)
        t = toa.get_TOAs([os.path.join(self.tim_directory,t) for t in toas],
                          usepickle=usepickle,
                          bipm_version=BIPM,
                          ephem=EPHEM,
                          planets=PLANET_SHAPIRO,
                          model=m,
                          picklefilename=picklefilename)

        # if we're dealing with wideband TOAs, each epoch has a single TOA, 
        # so don't bother checking to see if we can reduce entries
        if self.get_toa_type() == "NB":
            self.check_for_bad_epochs(t, threshold=0.9, print_all=print_all_ignores)

        # Add 'cut' flags to TOAs according to config 'ignore' block.
        t = self.apply_ignore(t)
        apply_cut_select(t,reason='configuration ignore block')

        # To facilitate TOA excision, frontend/backend info
        febe_pairs = set(t.get_flag_value('f')[0])
        log.info(f'Frontend/backend pairs present in this data set: {febe_pairs}')

        return m, t

    def get_bipm(self):
        """ Return the bipm string """
        if "bipm" in self.config.keys():
            return self.config['bipm']
        return None #return some default value instead?

    def get_ephem(self):
        """ Return the ephemeris string """
        if "ephem" in self.config.keys():
            return self.config['ephem']
        return None #return some default value instead?

    def print_changelog(self):
        """Print changelog entries from .yaml in the notebook."""
        # If there's a changelog, write out its contents. If not, complain.
        if 'changelog' in self.config.keys():
            print('changelog:')
            if self.config['changelog'] is not None:
                for cl in self.config['changelog']:
                    print(f'  - {cl}')
            else:
                print('...no changelog entries currently exist.')
        else:
            print('YAML file does not include a changelog. Add \'changelog:\' and individual entries there.')

    def get_fitter(self):
        """ Return the fitter string (do more?) """
        if "fitter" in self.config.keys():
            return self.config['fitter']
        return None

    def construct_fitter(self, to, mo):
        """ Return the fitter, tracking pulse numbers if available """
        fitter_name = self.config['fitter']
        fitter_class = getattr(pint.fitter, fitter_name)
        return fitter_class(to, mo)

    def get_toa_type(self):
        """ Return the toa-type string """
        if "toa-type" in self.config.keys():
            return self.config['toa-type']
        return None

    def get_niter(self):
        """ Return an integer of the number of iterations to fit """
        if "n-iterations" in self.config.keys():
            return int(self.config['n-iterations'])
        return 1

    def get_mjd_start(self):
        """Return mjd-start quantity (applies units days)"""
        if 'mjd-start' in self.config['ignore'].keys():
            return self.config['ignore']['mjd-start']*u.d
        return None

    def get_mjd_end(self):
        """Return mjd-end quantity (applies units days)"""
        if 'mjd-end' in self.config['ignore'].keys():
            return self.config['ignore']['mjd-end']*u.d
        return None

    def get_snr_cut(self):
        """ Return value of the TOA S/N cut """
        if "snr-cut" in self.config['ignore'].keys():
            return self.config['ignore']['snr-cut']
        return None #return some default value instead?

    def get_bad_epochs(self):
        """ Return list of bad epochs (basenames: [backend]_[mjd]_[source]) """
        bad_epoch_list = []
        if 'bad-epoch' in self.config['ignore'].keys() and self.config['ignore']['bad-epoch'] != None:
            for i in self.config['ignore']['bad-epoch']:
                if isinstance(i, list):
                    if len(i) == 2: # i.e. reason is provided, ignore it
                        bad_epoch_list.append([i[0]])
                    elif len(i) == 1: # no reason provided, but still a list
                        bad_epoch_list.append(i)
                elif isinstance(i, str): # still in old format
                    bad_epoch_list.append([i])
            return bad_epoch_list
        else:
            return None
        
    def get_bad_ranges(self):
        """ Return list of bad epoch ranges by MJD ([MJD1,MJD2])"""
        if 'bad-range' in self.config['ignore'].keys():
            return self.config['ignore']['bad-range']
        return None

    def get_bad_toas(self):
        """ Return list of bad TOAs (lists: [filename, channel, subint]) """
        bad_toa_list = []
        if 'bad-toa' in self.config['ignore'].keys() and self.config['ignore']['bad-toa'] != None:
            for i in self.config['ignore']['bad-toa']:
                if len(i) == 4: # i.e. reason is provided
                    bad_toa_list.append(i[:-1])
                elif len(i) == 3: # no reason provided
                    bad_toa_list.append(i)
            return bad_toa_list
        else:
            return None

    def get_investigation_files(self):
        """ Makes a list from which the timer can choose which epochs they'd like to manually inspect"""
        ff_list = sorted(glob.glob('/nanograv/timing/releases/15y/toagen/data/*/*/*.ff'))
        match_epochs, match_toas = [], []
        if 'bad-epoch' in self.config['ignore'].keys() and self.config['ignore']['bad-epoch'] != None:
            for be in self.get_bad_epochs():       
                match_epochs.append([filenm for filenm in ff_list if be[0] in filenm])
        if 'bad-toa' in self.config['ignore'].keys() and self.config['ignore']['bad-toa'] != None:
            for bt in self.get_bad_toas():
                match_toas.append([[filenm, bt[1], bt[2]] for filenm in ff_list if bt[0] in filenm])
        return sum(match_epochs,[]), sum(match_toas,[])

    def check_for_bad_epochs(self, toas, threshold=0.9, print_all=False):
        """Check the bad-toas entries for epochs where more than a given
        percentange of TOAs have been flagged. Make appropriate suggestions
        for the user to update the `bad-epoch` entires, and optionally
        supply the revised `bad-toa` entires.

        Parameters
        ----------
        toas: pint.TOA
            A PINT TOA object that contains a table of TOAs loaded

        threshold: float
            A threshold fraction used to determine whether to suggest adding
            a bad-epoch line to the config file. Should be in the range [0, 1].
            Default is 0.9.

        print_all: bool
            If True, print both the suggested bad-epoch lines AND the revised
            bad-toa lines, where the new bad-toa lines now have entries from
            the suggested bad-epochs removed. Default is False.
        """
        # get the list of bad-toas already in the config file
        # only continue if that list has entries
        provided_bad_toas = self.get_bad_toas()
        if isinstance(provided_bad_toas, list):
            bad_toa_epochs = np.asarray(provided_bad_toas)[:, 0]
            # how many bad TOAs per epoch?
            unique, counts = np.unique(bad_toa_epochs, return_counts=True)
            bad_toa_epoch_counts = dict(zip(unique, counts))

            # how many raw TOAs per epoch?
            toa_epochs = toas.get_flag_value("name")[0]
            unique, counts = np.unique(toa_epochs, return_counts=True)
            toa_epoch_counts = dict(zip(unique, counts))

            # get the list of bad-epochs already in the config
            provided_bad_epochs = self.get_bad_epochs()
            if not isinstance(provided_bad_epochs, list):
                provided_bad_epochs = []

            # are there any epochs that have too many bad TOAs?
            new_bad_epochs = []
            for k in bad_toa_epoch_counts:
                # at this point, TOAs could have already been removed,
                # so check that the key exists first
                if k in toa_epoch_counts.keys():
                    n_toas = toa_epoch_counts[k]
                    n_bad = bad_toa_epoch_counts[k]
                    bad_frac = float(n_bad) / n_toas
                    # check that the bad fraction exceeds the threshold
                    # AND that the current epoch isn't already listed
                    if bad_frac >= threshold and k not in provided_bad_epochs:
                        new_bad_epochs.append(k)

            # only bother printing anything if there's a suggestion
            if len(new_bad_epochs) > 0:
                log.warn(
                    f"More than {threshold * 100}% of TOAs have been excised for some epochs"
                )
                log.info("Consider adding the following to `bad-epoch` in your config file:")
                for e in new_bad_epochs:
                    print(f"    - '{e}'")

            # if requested to update the bad-toa lines, figure out which
            # entries need to be removed
            if print_all:
                all_bad_epochs = np.concatenate((new_bad_epochs, provided_bad_epochs))
                bad_toas_to_del = []
                for e in all_bad_epochs:
                    _idx = np.where(bad_toa_epochs == e)[0]
                    bad_toas_to_del.extend(_idx)
                new_bad_toa_list = np.delete(np.asarray(provided_bad_toas), bad_toas_to_del, 0)
                log.info("The `bad-toa` list in your config file can be reduced to:")
                for t in new_bad_toa_list:
                    print(f"    - ['{t[0]}',{t[1]},{t[2]}]")

    def get_prob_outlier(self):
        if "prob-outlier" in self.config['ignore'].keys():
            return self.config['ignore']['prob-outlier']
        return None #return some default value instead?

    def get_noise_dir(self):
        """ Return base directory for noise results """
        if 'results-dir' in self.config['noise'].keys():
            return self.config['noise']['results-dir']
        return None

    def get_ignore_dmx(self):
        """ Return ignore-dmx toggle """
        if 'ignore-dmx' in self.config['dmx'].keys():
            return self.config['dmx']['ignore-dmx']
        return None

    def get_fratio(self):
        """ Return desired frequency ratio """
        if 'fratio' in self.config['dmx'].keys():
            return self.config['dmx']['fratio']
        return FREQUENCY_RATIO

    def get_sw_delay(self):
        """ Return desired max(solar wind delay) threshold """
        if 'max-sw-delay' in self.config['dmx'].keys():
            return self.config['dmx']['max-sw-delay']
        return MAX_SOLARWIND_DELAY

    def get_custom_dmx(self):
        """ Return MJD/binning params for handling DM events, etc. """
        if 'custom-dmx' in self.config['dmx'].keys():
            return self.config['dmx']['custom-dmx']
        return None

    def apply_ignore(self,toas):
        """ Basic checks and return TOA excision info. """
        OPTIONAL_KEYS = ['mjd-start','mjd-end','snr-cut','bad-toa','bad-range','bad-epoch'] # prob-outlier, bad-ff
        EXISTING_KEYS = self.config['ignore'].keys()
        VALUED_KEYS = [k for k in EXISTING_KEYS if self.config['ignore'][k] is not None]

        # INFO?
        missing_valid = set(OPTIONAL_KEYS)-set(EXISTING_KEYS)
        if len(missing_valid):
            log.info(f'Valid TOA excision keys not present: {missing_valid}')

        invalid = set(EXISTING_KEYS) - set(OPTIONAL_KEYS)
        if len(invalid):
            log.warning(f'Invalid TOA excision keys present: {invalid}')

        valid_null = set(EXISTING_KEYS) - set(VALUED_KEYS) - invalid
        if len(valid_null):
            log.info(f'TOA excision keys included, but NOT in use: {valid_null}')

        valid_valued = set(VALUED_KEYS) - invalid
        if len(valid_valued):
            log.info(f'Valid TOA excision keys in use: {valid_valued}')

        # All info here about selecting various TOAs.
        # Select TOAs to cut, then use apply_cut_flag.
        if 'mjd-start' in valid_valued:
            start_select = (toas.get_mjds() < self.get_mjd_start()) # cut toas before mjd-start
            apply_cut_flag(toas,start_select,'mjdstart')
        if 'mjd-end' in valid_valued:
            end_select = (toas.get_mjds() > self.get_mjd_end()) # cut toas after mjd-end
            apply_cut_flag(toas,end_select,'mjdend')
        if 'snr-cut' in valid_valued:
            snr_select = ((np.array(toas.get_flag_value('snr')) < self.get_snr_cut())[0]) # cut toas below snr-cut
            apply_cut_flag(toas,snr_select,'snr')
            if self.get_snr_cut() > 8.0 and self.get_toa_type() == 'NB':
                log.warning('snr-cut should be set to 8; try excising TOAs using other methods.')
            if self.get_snr_cut() > 25.0 and self.get_toa_type() == 'WB':
                log.warning('snr-cut should be set to 25; try excising TOAs using other methods.')
        if 'prob-outlier' in valid_valued:
            pass
        if 'bad-ff' in valid_valued:
            pass
        if 'bad-epoch' in valid_valued:
            for be in self.get_bad_epochs():
                be_select = np.array([(be[0] in n) for n in toas.get_flag_value('name')[0]])
                apply_cut_flag(toas,be_select,'badepoch')
        if 'bad-range' in valid_valued:
            for br in self.get_bad_ranges():
                min_crit = (toas.get_mjds() > br[0]*u.d)
                max_crit = (toas.get_mjds() < br[1]*u.d)
                br_select = (min_crit & max_crit)
                # Look for backend (be) flag to further refine selection, if present
                if len(br) > 2:
                    be_select = np.array([(be == br[2]) for be in toas.get_flag_value('be')[0]])
                    br_select *= be_select
                apply_cut_flag(toas,br_select,'badrange')
        if 'bad-toa' in valid_valued:
            selection = np.zeros(len(toas),dtype=bool)
            for bt in self.get_bad_toas():
                name,chan,subint = bt
                name_match = np.array([(n == name) for n in toas.get_flag_value('name')[0]])
                chan_match = np.array([(ch == chan) for ch in toas.get_flag_value('chan')[0]])
                subint_match = np.array([(si == subint) for si in toas.get_flag_value('subint')[0]])
                if self.get_toa_type() == 'NB':
                    bt_select = (name_match * subint_match * chan_match)
                else:
                    # don't match based on -chan flags, since WB TOAs don't have them
                    bt_select = (name_match * subint_match)
                selection += bt_select
            apply_cut_flag(toas,selection,'badtoa')

        return toas