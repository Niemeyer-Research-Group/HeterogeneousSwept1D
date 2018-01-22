import os
import os.path as op
import sys
import pandas as pd
import numpy as np

thispath = op.abspath(op.dirname(__file__))
resultpath = op.dirname(thispath)
toppath = op.dirname(resultpath)
pypath = op.join(toppath, "runtools")
datapath = op.join(thispath, "rslts")

sys.path.append(pypath)

from main_help import *
import timing_analysis as ta
import timing_help as th

timeFrame = readPath(datapath)
coll = dict()
annodict = {}
pltpth = op.join(op.dirname(datapath), "AffinityPlots")

#idx = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=["Types", "Metric"])
bestCollect = pd.DataFrame(columns=list(timeFrame.keys()))
for kType, iFrame in timeFrame.items():
    thisdf = ta.RawInterp(iFrame, kType)
    
    keepdf = thisdf.interpit()
    dfT, figt, axT = ta.contourRaw(keepdf, kType, getfig=True)
    mnT = ta.plotmins(dfT, axT)
    figt = th.formatSubplot(figt)
    
    keepEfficiency = thisdf.efficient(keepdf)
    _, fige, _ = ta.contourRaw(keepEfficiency, kType, vals="efficiency", getfig=True)
    fige = th.formatSubplot(fige)
    
    bestCollect[kType] = mnT
    
    plotname = op.join(pltpth, "RawContour" + kType + "Time" + ".pdf")
    figt.savefig(plotname, bbox_inches='tight')
    plotname = op.join(pltpth, "RawContour" + kType + "Efficiency" + ".pdf")
    fige.savefig(plotname, bbox_inches='tight')
    
    # plt.show()

    

