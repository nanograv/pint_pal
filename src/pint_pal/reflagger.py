import pint.toa as toa
import sys

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
    
    
ipta_alt_flags = {'f': 'sys', 'fe': 'r', 'be': 'i'}

    
if  __name__=="__main__":
    narg = len(sys.argv)
    timfile = sys.argv[1]
    print('Tim file is', timfile)
    outfile = sys.argv[2]
    print('Outfile name is', outfile)
    pta = sys.argv[3]
    print('PTA is ', pta)
   
    
    process_reflagging(timfile, outfile, ipta_alt_flags, pta)
        
    

    
    

