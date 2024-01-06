"""Makes a movie from a simulated experiment with a control experiment. (Default movie created
    after a simulation does not include the control curve.)
    
    Args:
        <experiment pickle file> <control experiment pickle file (optional)>"""

import sys
import os
from simexp import SimReflExperiment, SimReflExperimentControl
from analysis import makemovie

assert (len(sys.argv) in [2, 3]), "syntax: python makemovie_control.py <exppickle> <ctrlpickle (optional)>"

exp = SimReflExperiment.load(sys.argv[1])
exppath = os.path.dirname(sys.argv[1])
expbase = os.path.basename(sys.argv[1])

if len(sys.argv) == 3:
    expctrl = SimReflExperimentControl.load(sys.argv[2])
else:
    expctrl = None

outfile = exppath + '/' + expbase.split('.pickle')[0] + '_ctrl'

print('Making movie %s...' % outfile)
makemovie(exp, outfile, expctrl=expctrl, fps=3, tscale='linear')




