"""
This code contains the TimingConfiguration class, which is used to load configuration files and perform
actions, with items then passed to the notebooks.

Very basic usage:
    from timingconfiguration import TimingConfiguration
    tc = TimingConfiguration(CONFIGFILE)
"""
import os
import pint.toa as toa
import pint.models as model
import numpy as np
import astropy.units as u
import yaml


class TimingConfiguration:
    """
    This class contains the functionality to read
    from a configuration file and send that information
    to the timing notebooks.
    """
    def __init__(self, filename="config.yaml"):
        """
        Initialization method

        Parameters
        ==========
        filename (optional) : path to the configuration file
        """
        self.filename = filename
        with open(filename) as FILE:
            self.config = yaml.load(FILE, Loader=yaml.FullLoader)


    def get_source(self):
        """ Return the source name """
        return self.config['source']

    def get_model(self):
        """ Return the PINT model object """
        filename = self.config["timing-model"]
        m = model.get_model(filename)
        if m.PSR.value != self.get_source():
            raise ValueError("%s source entry does not match parameter PSR"%self.filename)
        return m

    def get_freeparams(self):
        """Return list of free parameters"""
        return self.config['free-params']

    def get_TOAs(self):
        """ Return the PINT toa object """
        toas = self.config["toas"]
        tim_path = self.config["tim-directory"]
        BIPM = self.get_bipm()
        EPHEM = self.get_ephem() 
        
        # Individual tim file
        if isinstance(toas, str):
            tim_full_path = os.path.join(tim_path,toas)
            toas = toa.get_TOAs(tim_full_path, usepickle=False, bipm_version=BIPM, ephem=EPHEM)

        # List of tim files (currently requires writing temporary tim file with INCLUDE to read properly)
        elif isinstance(toas, list):
            source = self.get_source()
            tim_full_paths = [os.path.join(tim_path,t) for t in toas]
            f = open('TEMP.tim','w')
            for tf in tim_full_paths:
                f.write('INCLUDE %s\n' % (tf))
            f.close()
            toas = toa.get_TOAs('TEMP.tim', usepickle=False, bipm_version=BIPM, ephem=EPHEM)
            # Remove temporary tim file (TEMP.tim)?

        # Excise TOAs according to config 'ignore' block. 
        # (or should this happen as an explicit method run in case someone wants access to the raw TOAs?)
        self.apply_ignore(toas)

        return toas

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
        if 'bad-epoch' in self.config['ignore'].keys():
            return self.config['ignore']['bad-epoch']
        return None

    def get_bad_toas(self):
        """ Return list of bad TOAs (lists: [filename, channel, subint]) """
        if 'bad-toa' in self.config['ignore'].keys():
            return self.config['ignore']['bad-toa']
        return None

    def get_prob_outlier(self):
        if "prob-outlier" in self.config['ignore'].keys():
            return self.config['ignore']['prob-outlier']
        return None #return some default value instead?

    def apply_ignore(self,toas):
        """ Basic checks and return TOA excision info. """
        OPTIONAL_KEYS = ['mjd-start','mjd-end','snr-cut','prob-outlier','bad-ff','bad-epoch','bad-toa'] 
        EXISTING_KEYS = self.config['ignore'].keys()
        VALUED_KEYS = [k for k in EXISTING_KEYS if self.config['ignore'][k] is not None]

        # INFO?
        missing_valid = set(OPTIONAL_KEYS)-set(EXISTING_KEYS)
        # WARNING?
        invalid = set(EXISTING_KEYS) - set(OPTIONAL_KEYS)
        # INFO?
        valid_null = set(EXISTING_KEYS) - set(VALUED_KEYS) - invalid
        # INFO?
        valid_valued = set(VALUED_KEYS) - invalid

        selection = np.ones(len(toas),dtype=bool)

        # All info here about selecting various TOAs.
        if 'mjd-start' in valid_valued:
            start_select = (toas.get_mjds() > self.get_mjd_start())
            selection *= start_select
        if 'mjd-end' in valid_valued:
            end_select = (toas.get_mjds() < self.get_mjd_end())
            selection *= end_select
        if 'snr-cut' in valid_valued:
            snr_select = ((np.array(toas.get_flag_value('snr')) > self.get_snr_cut())[0])
            selection *= snr_select
        if 'prob-outlier' in valid_valued:
            pass
        if 'bad-ff' in valid_valued:
            pass
        if 'bad-epoch' in valid_valued:
            for be in self.get_bad_epochs():
                be_select = np.array([(be not in n) for n in toas.get_flag_value('name')[0]])
                selection *= be_select 
        if 'bad-toa' in valid_valued:
            for bt in self.get_bad_toas():
                name,chan,subint = bt
                name_match = np.array([(n == name) for n in toas.get_flag_value('name')[0]])
                chan_match = np.array([(ch == chan) for ch in toas.get_flag_value('chan')[0]])
                subint_match = np.array([(si == subint) for si in toas.get_flag_value('subint')[0]])
                bt_select = np.invert(name_match * subint_match * chan_match)
                selection *= bt_select

        toas.select(selection)
