'''
    Classes and functions for plotting the actual results of the simulations.
    Interpolating version
'''

# Dependencies: gitpython,cycler

import collections
import itertools
import json
import os
import os.path as op
import sys

#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
 
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


sty = op.join(thispath, "swept.mplstyle")
plt.style.use(sty)

hxColors = ["#{:02x}{:02x}{:02x}".format(r,g,b) for r, g, b in myColors]

plt.rc('axes', prop_cycle=cycler('color', hxColors) + cycler('marker', ['D', 'o', 'h', '*', '^', 'x', 'v', '8']))

xlbl = "Grid Size"

dftype=pd.core.frame.DataFrame #Compare types

schemes = ["Classic", "Swept"]

schs = dict(zip([k[0] for k in schemes], schemes))

meas = {"time": "time per timestep (us)", "efficiency": "MGridPts/s", "spd": "Speedup", "spdg":"Speedup vs GPU", "btpb":"Best tpb comparison"}

fc = {'time':'min', 'efficiency': 'max'}

def rowSelect(df, n):
    return df.iloc[::len(df.index)//n, :]

def formatSubplot(f):
    nsub = len(f.axes)
    if nsub == 4:
        f.tight_layout(pad=0.2, w_pad=0.75, h_pad=0.75)
        f.subplots_adjust(top=0.9, bottom=0.08, right=0.85, hspace=0.3, wspace=0.3)
    if nsub > 4:
        f.tight_layout(pad=0.2, w_pad=0.75, h_pad=0.75)
        f.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3, wspace=0.3)

    return f

def cartProd(y):
    return pd.DataFrame(list(itertools.product(*y))).values

def nxRange(mmax, n):
    rng=n//10 
    return np.around(np.logspace(*(np.log2(mmax['nX'])), base=2, num=n), decimals=1)[rng:-rng]

def simpleNGrid(uniq, mmax):
    xint = [uniq['tpb'], uniq['gpuA']]
    xint.append(nxRange(mmax, 100))
    return xint

# At this point the function is independent
def getBestAll(df, respvar):
    f = fc[respvar]
    ifx = 'idx'+f
    ti = list(df.columns.names)
    ti.remove('metric')
    ixvar = list(ti)[0]
    to = list(df.columns.get_level_values('metric').unique())
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
    def __init__(self, df, name):
        self.oFrame = df
        self.title = name
        self.cols = list(df.columns.values)
        self.nobs = len(df)
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
            kwarg = {'loglog': True}
        else:
            kwarg = {'logx': True}

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


def addIntercept(perf):
    iprod = cartProd(perf.uniques['tpb'], perf.uniques['gpuA'], [0.0], [0.0])
    zros = pd.DataFrame(iprod, columns=perf.cols)
    return pd.concat(perf.oFrame, zros)
