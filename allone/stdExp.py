'''
    Run experiment.

'''

import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import subprocess as sp
import shlex

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
toppath = op.dirname(thispath)
pypath = op.join(toppath, "runtools")
binpath = op.join(thispath, "bin")
resultpath = op.join(thispath, "results")

sys.path.append(pypath)
import result_help as rh
import main_help as 

''' STUFF: Not really makeList, make iterable '''

prog = op.join(binpath, 'euler')

mh.runMPICUDA()