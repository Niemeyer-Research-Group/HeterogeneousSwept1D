'''
    Run the performance test described in the paper.  
    Will save all best runs to an hd5 file in a pandas dataframe in Results folder.
    Will also save ALL timing for the last run to appropriately named text files in Results folder.
'''

import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from main_help import *
import timing_help as th
from sklearn import linear_model
import re

schemes = {"C": "Classic", "S": "Swept"}

tfiles = sorted([k for k in os.listdir(rspath) if k.startswith('t')])

res = []
ti = []
for tf in tfiles:
    pth = op.join(rspath, tf) 
    opt = re.findall('[A-Z][^A-Z]*', tf)
    ti.append(opt[0]+schemes[opt[1][0]])
    res.append(th.parseCsv(pth))
   
hdfpath = op.join(resultpath, "rawResults.h5")    
th.longTerm(res, ti, hdfpath)
