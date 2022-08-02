import pint.toa as toa

# This is v0 of a script to load in a tim file and bring flags into alignment with the IPTA data standard
# TO-DO
# ------
# -command line set-up so that I'm not making name & tim file changes in the script
# -Find edge cases
# -make better organized/less adhoc.


tim_file = 'tims/J1909-3744/NRT.BON.1400.tim'
bipm_ver = 'BIPM2019'
eph = 'DE441'

t = toa.get_TOAs(tim_file, bipm_version=bipm_ver,ephem=eph)

# Ensure every tim has a value for f. 
# Match to sys if available, otherwise group.


if t.get_flag_value('f')[0][0] == None:
    if t.get_flag_value('sys')[0] != None:
        t['f'] = t['sys'][0]
    elif t.get_flag_value('group')[0] !=None:
        t['f'] = t['group'][0]
    else:
        t['f'] = 'unknown'
        

# fix receiver
alt_fe = 'r'
if t.get_flag_value('fe')[0][0] == None:
    if t.get_flag_value(alt_fe)[0][0] != None:
        t['fe'] = t[alt_fe][0]
    else:
        t['fe'] = 'unknown'

# alterate 'be option'
alt_be = 'i'
if t.get_flag_value('be')[0][0] == None:
    if t.get_flag_value(alt_be)[0][0]  == None:
        t['be'] = t[alt_be][0]
    else:
        t['be'] = 'unknown'
        
        
# fix other flags
if t.get_flag_value('proc')[0][0] == None:
    t['proc'] = 'unknown'
if t.get_flag_value('B')[0][0] == None:
    t['B'] = 'unknown'
if t.get_flag_value('bw')[0][0] == None:
    t['bw'] = 'unknown'
if t.get_flag_value('tobs')[0][0] == None:
    t['tobs'] = 'unknown'
if t.get_flag_value('pta')[0][0] == None:
    t['pta'] = 'unknown'
    
    
# Write new par
outfile = 'reflagged_test.tim'
t.write_TOA_file(outfile, format='tempo2')
