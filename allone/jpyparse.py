"""
    Parse the json and make some output.
"""
import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
import shlex

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
toppath = op.dirname(thispath)
pypath = op.join(toppath, "runtools")
binpath = op.join(thispath, "bin")
resultpath = op.join(toppath, "results")

sys.path.append(pypath)
import result_help as rh
import main_help as mh


f = op.join(thispath, "solutes.json")
jdf = rh.Solved(f)
fg, axi = plt.subplots(1, 1)
jdf.plotResult(fg, axi)
jdf.savePlot(fg, resultpath)
    
                