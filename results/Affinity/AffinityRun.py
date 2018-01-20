import os
import os.path as op
import sys
import pandas as pd
import numpy as np

thispath = op.abspath(op.dirname(__file__))
resultpath = op.dirname(thispath)
toppath = op.dirname(resultpath)
pypath = op.join(toppath, "runtools")
datapath = op.join(thispath, "tests")

sys.path.append(pypath)

import timing_analysisA as ta

if __name__ == "__main__":
    fy = [op.join(datapath, k) for k in os.listdir(datapath) if k.endswith('.csv')]
    coll = dict()
    for fa in fy:
        op. split dir
        thisdf = ta.RawInterp(pd.read_csv(fa), getname)
        thisdf.interpit()
        # Gather
        ta.plotContour(ndf, axi, annot)

    

