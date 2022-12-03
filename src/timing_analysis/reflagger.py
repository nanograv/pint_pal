#!/usr/bin/env python
import argparse
import asyncio

import pathlib
import pint.logging
import pint.toa as toa
import subprocess
import sys
import tomli

from loguru import logger as log
#pint.logging.setup(level=pint.logging.script_level)


# the orignal ###############################################################
#
# This is v0 of a script to load in a tim file and bring flags into alignment with the IPTA data standard
# TO-DO
# ------
# -Find edge cases

def reflag_add_empty(toa, flag_name):
    if toa.get_flag_value(flag_name)[0][0] == None:
        toa[flag_name] = 'unknown'

def reflag_add_value(toa, flag_name, value):
    if toa.get_flag_value(flag_name)[0][0] == None:
        toa[flag_name] = value

def reflag_alt_name(toa, flag_name, alt_name):
    if toa.get_flag_value(flag_name)[0][0] == None:
        print(flag_name, 'not present')
        if toa.get_flag_value(alt_name)[0][0] == None:
            toa[flag_name] = 'unknown'
#            print(alt_name,'not present')
        else:
            toa[flag_name] = toa[alt_name][0]
#            print(alt_name, 'is present')


def ipta_standard_reflag(toa, f_alt, fe_alt, be_alt, pta):
    # Account for alternative names for fe/be/f
    reflag_alt_name(toa, 'f', f_alt)
    reflag_alt_name(toa, 'fe', fe_alt)
    reflag_alt_name(toa, 'be', be_alt)

    # Add value if flag is missing but the value is known.
    reflag_add_value(toa, 'pta', pta)

    # Add 'unknown' if flags are missing (where there is not a known alternative)
    reflag_add_empty(toa, 'proc')
    reflag_add_empty(toa, 'B')
    reflag_add_empty(toa, 'bw')
    reflag_add_empty(toa, 'tobs')
    #reflag_add_empty(toa, 'fe')


def process_reflagging(timfile, outfile, alt_flag_dict, pta, bipm_ver='BIPM2019', eph='DE440'):
    t = toa.get_TOAs(timfile, bipm_version=bipm_ver,ephem=eph)
    ipta_standard_reflag(t, alt_flag_dict['f'], alt_flag_dict['fe'], alt_flag_dict['be'], pta)

    t.write_TOA_file(outfile, format='tempo2')

#############################################################################


# contains the logic for the new version
class HoldingClass():
    def process_reflagging(self, toa):
        for action, args_l in self.__operations_l:
            self.__actions_d[action](toa, *args_l)


    # both flags set in the data originally is normal and okay?
    def action_clone(self, toa, source, dest):
        both = [
            toa.get_flag_value(source)[0],
            toa.get_flag_value(dest)[0]
        ]
        a,b = both

        flags = [
            a[i]
            if a[i] is not None
            else b[i] if b[i] is not None
            else self.__missing_flag
            for i in range(len(a))
        ]

        toa[source] = flags
        toa[dest] = flags


    # need to accommodate the case where a variable is referred to
    def action_add_if_missing(self, toa, flag, value):
        if toa.get_flag_value(flag)[0][0] is None:
            toa[flag] = (
                locals()[value] if value in locals() else self.__missing_flag
            )

    def __init__(self, operations_l, missing_flag):
        self.__operations_l = operations_l
        self.__missing_flag = missing_flag

        self.__actions_d = {
            'clone' : self.action_clone,
            'add_if_missing' : self.action_add_if_missing
        }

        # legacy
        self.ipta_alt_flags = {'f': 'sys', 'fe': 'r', 'be': 'i'}


def git_diff_color_words(orig_file_path, new_file_path):
    diff_output = subprocess.Popen(
        [
            'git',
            '--no-pager',
            'diff',
            '--color-words',
            '--no-index',
            orig_file_path,
            new_file_path
        ]
    )
    print(diff_output)


async def main(
       *,
       config_file_path = None,
       tim_file_path = None,
       output_file_path = None,
       pulsar_timing_array = None,
       do_diff = False
):

    # use async here for parallel file i/o if helpful
    _config_d = tomli.load(open(config_file_path, 'rb'))

    potatoes = HoldingClass(
        operations_l = _config_d['actions'],
        missing_flag = _config_d['defaults']['missing_flag']
    )

    # legacy
    process_reflagging(
        tim_file_path,
        output_file_path,
        potatoes.ipta_alt_flags,
        pulsar_timing_array
    )

    # new
    t = toa.get_TOAs(
        tim_file_path,
        bipm_version = _config_d['defaults']['bipm'],
        ephem = _config_d['defaults']['ephemeris']
    )
    potatoes.process_reflagging(t)
    t.write_TOA_file(f'{output_file_path}.new', format='tempo2')

    # compare (lol, just run diff and print the output)
    if do_diff:
        diff_output = git_diff_color_words(
            #output_file_path,
            tim_file_path,
            f'{output_file_path}.new'
        )
        print(diff_output)


def entry_point(*, args = None):
    asyncio.run(
        main(
            config_file_path = args.config_file_path[0],
            tim_file_path = args.tim_file_path[0],
            output_file_path = args.output_file_path[0],
            pulsar_timing_array = args.pulsar_timing_array[0],
            do_diff = args.do_diff
        )
    )


if  __name__=="__main__":
    desc = 're-flag a Tim file for IPTA combined analyses'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--config',
        '-c',
        nargs = 1,
        dest = 'config_file_path',
        type = pathlib.Path,
        help = 'the toml file defining the flag mapping',
        default = None,
        required = True
    )

    parser.add_argument(
        '--timfile',
        '-t',
        nargs = 1,
        dest = 'tim_file_path',
        type = pathlib.Path,
        help = 'the Tim file to re-flag',
        default = None,
        required = True
    )

    parser.add_argument(
        '--outputfile',
        '-o',
        nargs = 1,
        dest = 'output_file_path',
        type = pathlib.Path,
        help = 'path to the output file',
        default = None,
        required = True
    )

    parser.add_argument(
        '--pta',
        '-p',
        nargs = 1,
        dest = 'pulsar_timing_array',
        type = str,
        help = 'the PTA to which the orignal Tim file belongs',
        default = None,
        required = True
    )

    parser.add_argument(
        '--diff',
        '-d',
        dest = 'do_diff',
        help = 'output a colorized diff between the two versions',
        action='store_const',
        const=True,
        default = False
    )

    args = parser.parse_args()

    log.info(f"Using mapping file '{args.config_file_path[0]}'.")

    log.info(
        f"Re-flagging '{args.tim_file_path[0]}' from PTA "
        f"'{args.pulsar_timing_array[0]}' and writing to "
        f"'{args.output_file_path[0]}'."
    )

    entry_point(args = args)
