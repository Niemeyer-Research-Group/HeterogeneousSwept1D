'''
    Classes and functions for plotting the actual results of the simulations.

'''


import os
import sys
import os.path as op
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
import pandas as pd
import palettable.colorbrewer as pal
import collections

from main_help import *
import result_help as rh

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

ext = ".json"

def parseJdf(fb):
    if isinstance(fb, str):
        jdict = readj(fb)
    elif isinstance(fb, dict):
        jdict=fb
        
    return jdict
    
def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def swapKeys(d):
    b = collections.defaultdict(dict)
    for k0 in d.keys():
        for k1 in d[k0].keys():
            if d[k0][k1]:
                b[k1][k0] = d[k0][k1]

    return b

class Perform(object):
    
    def __init__(self, dataset):
        self.oDict = parseJdf(dataset) #Takes either dict or path to json.
        self.dpth = depth(self.oDict)
        self.dFrame = dictframes(self.oDict, self.dpth)

    def plotdict(self, eqName, plotpath=".", saver=True, shower=False, swap=False):
        for k0 in self.oDict.keys():
            ptitle = eqName + " | tpb = " + k0
            plotname = op.join(plotpath, eqName + k0 + ".pdf")
            if swap:
                self.nDict = swapKeys(self.oDict[k0])
            else:
                self.nDict = self.oDict[k0]
            lg = sorted(self.nDict.keys())
            lg = lg[1::2]
            for k1 in lg:
                x = []
                y = []
                lo = [int(k) for k in self.nDict[k1].keys()]
                x = sorted(lo)
                for k2 in x:
                    y.append(self.nDict[k1][str(k2)])
                    

                print(x, y)
                plt.loglog(x, y, linewidth=2, label=k1)
            plt.grid(True)
            plt.title(ptitle)
            plt.ylabel("Time per timestep (us)")
            plt.xlabel("Grid Size")
            plt.legend(lg)
            if saver:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="GPU Affinity", borderaxespad=0.)
                plt.savefig(plotname, dpi=1000, bbox_inches="tight")
            if shower:
                plt.show()
        
    def plotframe(self, eqName, plotpath=".", saver=True, shower=False):
        for ky in self.dFrame.keys():
            ptitle = eqName + " | tpb = " + ky
            plotname = op.join(plotpath, eqName + ky + ".pdf")
        
            self.dFrame[ky].plot(logx = True, logy=True, grid=True, linewidth=2, title=ptitle)
            plt.ylabel("Time per timestep (us)")
            plt.xlabel("Grid Size")
            if saver:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="GPU Affinity", borderaxespad=0.)
                plt.savefig(plotname, dpi=1000, bbox_inches="tight")
            if shower:
                plt.show()
    

class RunParse(object):
    def __init__(self, dataMat):
        self.bestRun = pd.DataFrame(dataMat.min(axis=1))
        bestIdx = dataMat.idxmin(axis=1)
        self.bestLaunch = bestIdx.value_counts()
        self.bestLaunch.index = pd.to_numeric(self.bestLaunch)
        self.bestLaunch.sort_index(inplace=True)

    
class Perform1(object):
    
    def __init__(self, oDict):
        self.oDict = undict(oDict)

    def plotdict(self, eqName, plotpath=".", saver=True, shower=False, swap=False):
        for k0 in self.oDict.keys():
            ptitle = eqName + " | tpb = " + k0
            plotname = op.join(plotpath, eqName + k0 + ".pdf")
            if swap:
                self.nDict = swapKeys(self.oDict[k0])
            else:
                self.nDict = self.oDict[k0]
            lg = sorted(self.nDict.keys())
            lg = lg[1::2]
            for k1 in lg:
                x = []
                y = []
                lo = [int(k) for k in self.nDict[k1].keys()]
                x = sorted(lo)
                for k2 in x:
                    y.append(self.nDict[k1][str(k2)])
                    

                print(x, y)
                plt.loglog(x, y, linewidth=2, label=k1)
            plt.grid(True)
            plt.title(ptitle)
            plt.ylabel("Time per timestep (us)")
            plt.xlabel("Grid Size")
            plt.legend(lg)
            if saver:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="GPU Affinity", borderaxespad=0.)
                plt.savefig(plotname, dpi=1000, bbox_inches="tight")
            if shower:
                plt.show()
        
    def plotframe(self, eqName, plotpath=".", saver=True, shower=False):
        for ky in self.dFrame.keys():
            ptitle = eqName + " | tpb = " + ky
            plotname = op.join(plotpath, eqName + ky + ".pdf")
        
            self.dFrame[ky].plot(logx = True, logy=True, grid=True, linewidth=2, title=ptitle)
            plt.ylabel("Time per timestep (us)")
            plt.xlabel("Grid Size")
            if saver:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="GPU Affinity", borderaxespad=0.)
                plt.savefig(plotname, dpi=1000, bbox_inches="tight")
            if shower:
                plt.show()


def plotItBar(axi, dat):

    rects = axi.patches
    for r, val in zip(rects, dat):
        axi.text(r.get_x() + r.get_width()/2, val+.5, val, ha='center', va='bottom')

    return

class QualityRuns(object):
    def __init__(self, dataMat):
        self.bestRun = pd.DataFrame(dataMat.min(axis=1))
        bestIdx = dataMat.idxmin(axis=1)
        self.bestLaunch = bestIdx.value_counts()
        self.bestLaunch.index = pd.to_numeric(self.bestLaunch)
        self.bestLaunch.sort_index(inplace=True)

## Make a generalized linear model out of this
def glmRun():
    pass

if __name__ == "__main__":
    fnow = input("input the file-name without extension ")
    f = fnow + ext
    ff = find_all(f, "..")
    pobj = Perform(ff[0])
    pobj.plotframe("Euler", saver=False, shower=True)


#def normJdf(fpath, xlist):
#    jdict = parseJ(fpath)
#    jd = dict()
#    for jk in jdict.keys():
#
#        jd[jk] = dict()
#        for jkk in jdict[jk].keys():
#            jknt, idx = min([(abs(int(jkk)-x), i) for i, x in enumerate(xlist)])
#            
#            jkn = xlist[idx]
#            print(jknt, idx, jkk, jkn)
#            
#            if jkn not in jd[jk].keys():
#                jd[jk][jkn] = dict()
#                
#                
#            jd[jk][jkn].update(jdict[jk][jkk])
#        
#            
#    return jd
#
#def nicePlot(df, ptitle, ltitle):
#    df.plot(grid=True, logy=True, title=ptitle)
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title=ltitle, borderaxespad=0.)
#    #Saveit as pdf?    
#    return True

    
