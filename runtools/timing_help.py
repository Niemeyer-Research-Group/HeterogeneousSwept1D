'''
    Classes and functions for plotting the actual results of the simulations.
    Interpolating version
'''

# Dependencies: gitpython, palettable, cycler

import collections
import itertools
import json
import os
import os.path as op
import sys
from datetime import datetime

import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
import re

from scipy import interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf
 

from main_help import *

plt.close('all')

myColors = [[27, 158, 119],
 [217, 95, 2],
 [117, 112, 179],
 [231, 41, 138],
 [102, 166, 30],
 [230, 171, 2],
 [166, 118, 29],
 [102, 102, 102]]

hxColors = ["#{:02x}{:02x}{:02x}".format(r,g,b) for r, g, b in myColors]

plt.rc('axes', prop_cycle=cycler('color', hxColors))
    #+cycler('marker', ['D', 'o', 'h', '*', '^', 'x', 'v', '8']))

mpl.rcParams['lines.linewidth'] = 3
#mpl.rcParams['lines.markersize'] = 8
mpl.rcParams["figure.figsize"] = 14,8
mpl.rcParams["figure.titlesize"]="x-large"
mpl.rcParams["figure.titleweight"]="bold"
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True
mpl.rcParams["savefig.dpi"] = 1000
mpl.rcParams["savefig.bbox"] = "tight"

xlbl = "Grid Size"

dftype=pd.core.frame.DataFrame #Compare types

schemes = ["Classic", "Swept"]

schs = dict(zip([k[0] for k in schemes], schemes))

meas = {"time": "us per timestep", "efficiency": "MGridPts/s", "SpeedupAlg":"Speedup", "SpeedupGPU":"Speedup", "Best tpb":"Best tpb comparison"}

fc = {'time':'min', 'efficiency': 'max'}

def rowSelect(df, n):
    return df.iloc[::len(df.index)//n, :]

def todayDate():
    return datetime.date(datetime.today).isoformat().replace("-", "_")

def formatSubplot(f):
    nsub = len(f.axes)
    if nsub == 4:
        f.tight_layout(pad=0.2, w_pad=0.75, h_pad=0.75)
        f.subplots_adjust(top=0.9, bottom=0.08, right=0.85, hspace=0.3, wspace=0.3)
        
    return f

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

def parseCsv(fb):
    if isinstance(fb, str):
        jframe = pd.read_csv(fb)
    elif isinstance(fb, dftype):
        jframe = fb

    print(jframe)
    jframe = jframe[(jframe.nX !=0)]

    return jframe

def cartProd(y):
    return pd.DataFrame(list(itertools.product(*y))).values

# Takes list of dfs? title of each df, longterm hd5, option to overwrite
# incase you write wrong.  Use carefully!
def longTerm(dfs, titles, fhdf, overwrite=False):
    today = str(datetime.date(datetime.today()))
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    nList = []
    for d, t in zip(dfs, titles):
        d["eqs"] = [t]*len(d)
        nList.append(d.set_index("eqs"))

    dfcat = pd.concat(nList)

    print(len(dfcat))

    opStore = pd.HDFStore(fhdf)
    fnote = op.join(op.dirname(fhdf), "notes.json")
    if op.isfile(fnote):
        dnote = readj(fnote)
    else:
        dnote = dict()

    if sha in dnote.keys() and not overwrite:
        opStore.close()
        print("You need a new commit before you save this again")
        return "Error: would overwrite previous entry"

    dnote[sha] = {"date": today}
    dnote[sha]["System"] = input("What machine did you run it on? ")
    dnote[sha]["np"] = int(input("Input # MPI procs? "))
    dnote[sha]["note"] = input("Write a note for this data save: ")

    with open(fnote, "w") as fj:
        json.dump(dnote, fj)

    opStore[sha] = dfcat
    opStore.close()
    return dfcat

# At this point the function is independent
def getBestAll(df, respvar):
    f = fc[respvar]
    ifx = 'idx'+f
    ti = list(df.columns.names)
    ti.remove('metric')
    ixvar = list(ti)[0]
    print(ixvar)
    to = list(df.columns.get_level_values('metric').unique())
    print(to)
    to.remove(respvar)
    oxvar = to[0]

    dfb = pd.DataFrame(df[respvar].apply(f, axis=1), columns=[respvar])
    dfb[ixvar] = pd.DataFrame(df[respvar].apply(ifx, axis=1))

    bCollect = []
    for i, k in zip(dfb.index, dfb[ixvar].values):
        bCollect.append(df[oxvar].loc[i, k])

    dfb[oxvar] = bCollect
    return dfb

class Perform(object):
    def __init__(self, df, name, icept=False):
        self.oFrame = df
        self.title = name
        self.cols = list(df.columns.values)
        self.xo = self.cols[:3]
        self.uniques, self.minmaxes = {}, {}
        self.iFrame = pd.DataFrame()
        self.bFrame = pd.DataFrame()

        for k in self.xo:
            tmp = self.oFrame[k].unique()
            self.uniques[k] = tmp
            self.minmaxes[k] = [tmp.min() , tmp.max()]

        self.minmaxes['nX'] = [self.oFrame.groupby(self.xo[:2]).min()['nX'].max(), 
                                self.oFrame.groupby(self.xo[:2]).max()['nX'].min()]
                
        if icept:
            addIntercept()

    def __str__(self):
        ms = "%s \n %s \n Unique Exog: \n" % (self.title, self.oFrame.head())

        for i, s in enumerate(self.uniques):
            ms = ms + self.cols[i] + ": \n " + str(s) + "\n"
        return ms

    def efficient(self, df=pd.DataFrame()):
        if not df.empty:
            df['efficiency'] = df['nX']/df['time']
            return df
        
        self.oFrame['efficiency'] = self.oFrame['nX']/self.oFrame['time']
        if not self.iFrame.empty:
            self.iFrame['efficiency'] = self.iFrame['nX']/self.iFrame['time']

    def plotRaw(self, subax, respvar, legstep=1):
        legax = self.xo[1] if subax==self.xo[0] else self.xo[0] 
        
        drops = self.uniques[legax][1::legstep]
        saxVal = self.uniques[subax]
        ff = []
        ad = {}
        if respvar=='time':
            kwarg = {'loglog':True}
        else:
            kwarg = {'logx':True}

        for i in range(len(saxVal)//4):
            f, ai = plt.subplots(2,2)
            ap = ai.ravel()
            ff.append(f)
            for aa, t in zip(ap, saxVal[i::2]):
                ad[t] = aa

        for k, g in self.oFrame.groupby(subax):
            for kk, gg in g.groupby(legax):
                if kk in drops:
                    gg.plot(x='nX', y=respvar, ax=ad[k], grid=True, label=kk, **kwarg)
            
            ad[k].set_title(k)
            ad[k].set_ylabel(meas[respvar])
            ad[k].set_xlabel(xlbl)
            hd, lb = ad[k].get_legend_handles_labels()
            ad[k].legend().remove()

        for i, fi in enumerate(ff):
            fi = formatSubplot(fi)
            fi.legend(hd, lb, 'upper right', title=legax, fontsize="medium")
            fi.suptitle(self.title + " - Raw " +  respvar +  " by " + subax)

        return ff

    def getBest(self, subax, respvar, df=pd.DataFrame()):
        f = fc[respvar]
        ifx = 'idx'+f
        legax = self.xo[1] if subax==self.xo[0] else self.xo[0] 
        blankidx = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['metric', subax])
        fCollect = pd.DataFrame(columns=blankidx)

        if df.empty:
            df = self.iFrame

        for k, g in df.groupby(subax):
            gg = g.pivot(index='nX', columns=legax, values=respvar)
            fCollect[respvar, k] = gg.apply(f, axis=1)
            fCollect[legax, k] = gg.apply(ifx, axis=1)

        return fCollect

    def nxRange(self, n):
        rng=n//10 
        return np.around(np.logspace(*(np.log2(self.minmaxes['nX'])), base=2, num=n), decimals=1)[rng:-rng]

    def simpleNGrid(self):
        xint = [self.uniques['tpb'], self.uniques['gpuA']]
        xint.append(self.nxRange(100))
        return xint

    def addIntercept(self):
        iprod = cartProd(self.uniques['tpb'], self.uniques['gpuA'], [0.0], [0.0])
        zros = pd.DataFrame(iprod, columns=self.cols)
        self.oFrame = pd.concat(self.oFrame, zros)

class PerformFilter(pd.DataFrame):
    def __init__(self, df):
        pass
    
    def oneKindofFilter(self):
        pass