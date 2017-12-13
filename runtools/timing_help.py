'''
    Classes and functions for plotting the actual results of the simulations.

'''

# Dependencies: gitpython, palettable, cycler

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
import git
import json
import statsmodels.api as sm
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import itertools
from random import random

from main_help import *
import result_help as rh

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

ylbl = "Time per timestep (us)"
xlbl = "Grid Size"
ext = ".json"

schemes = {"C": "Classic", "S": "Swept"}

cols = ["tpb", "gpuA", "nX", "time"]

crs=["r", "b", "k", "g"]

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

def parseCsv(fb):
    if isinstance(fb, str):
        jframe = pd.read_csv(fb)
    elif isinstance(fb, pd.core.frame.DataFrame):
        jframe = fb

    jframe.columns = cols
    jframe = jframe[(jframe.nX !=0)]

    return jframe


class RunParse(object):
    def __init__(self, dataMat):
        self.bestRun = pd.DataFrame(dataMat.min(axis=1))
        bestIdx = dataMat.idxmin(axis=1)
        self.bestLaunch = bestIdx.value_counts()
        self.bestLaunch.index = pd.to_numeric(self.bestLaunch)
        self.bestLaunch.sort_index(inplace=True)
        

#takes list of dfs? title of each df, longterm hd5, option to overwrite incase you write wrong.  Use carefully!
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


def xinterp(dfn):
    gA = np.linspace(min(dfn.gpuA), max(dfn.gpuA), 50)
    nXA = np.linspace(min(dfn.nX), max(dfn.nX), 50)

    # There should be a combinatorial function.
    return np.array(np.meshgrid(gA, nXA)).T.reshape(-1,2)


class Perform(object):
    def __init__(self, df, name):
        self.oFrame = df
        self.title = name
        self.cols = list(df.columns.values)
        self.uniques, self.minmaxes = [], []
            
        for k in self.cols[:-1]:
            self.uniques.append(list(np.unique(self.oFrame[self.cols[0]])))
            self.minmaxes.append( (min(self.uniques[-1]), max(self.uniques[-1])) ) 

    def glmBytpb(self, mdl):
        iframe = self.oFrame.set_index("tpb")
        self.gframes = dict()
        self.gR2 = dict()
        for th in self.uniques[0]:
            dfn = iframe.xs(th)
            mdl.fit(dfn[cols[1:-1]], dfn[cols[-1]])
            xi = xinterp(dfn)
            yi = mdl.predict(xi)
            xyi = pd.DataFrame(np.vstack((xi.T, yi.T)).T, columns=cols[1:])

            self.gframes[th] = xyi.pivot(*self.cols[1:])
            self.gR2[th] = r2_score(dfn[cols[-1]], mdl.predict(dfn[cols[1:-1]]))

        mnmx = np.array(minmaxes)
        self.lims = np.array((mnmx[:,:2].max(axis=0), mnmx[:,-2:].min(axis=0)))
        #self.lims = np.append(b, minmaxes[-2:].min(axis=1).values)

        return mdl

    def modelGeneral(self, mdl):
        mdl.fit(self.oFrame[cols[:-1]], self.oFrame[cols[-1]])
        return mdl

    def modelSm(self):
        mdl = sm.RecursiveLS(self.oFrame[cols[-1]], self.oFrame[cols[:-1]]).fit()
        mdl.summary()
        return mdl

    def useModel(self, mdl):
        mdl = modelGeneral(mdl)
        xInt = xinterp(self.oFrame)
        yInt = mdl.predict(xInt)
        self.pFrame = pd.DataFrame(np.vstack((xInt.T, yInt.T)).T, columns=cols)
        


    def transformz(self):
        self.ffy = self.pFrame.set_index(cols[0]).set_index(cols[1], append=True)
        self.fffy = self.ffy.set_index(cols[2], append=True)
        self.ffyo = self.fffy.unstack(cols[2])
        self.pMinFrame = pd.DataFrame(self.ffyo.min().unstack(0))
        self.pMinFrame.columns = ["time"]

    def plotframe(self, plotpath=".", saver=True, shower=False):
        iframe = self.oFrame.set_index("tpb")
        for ky in self.tpbs:
            ptitle = self.title + " | tpb = " + ky
            plotname = op.join(plotpath, self.title + ky + ".pdf")
            ifx = iframe.xs(ky).pivot(*self.cols[1:])

            ifx.plot(logx = True, logy=True, grid=True, linewidth=2, title=ptitle)
            plt.ylabel(ylbl)
            plt.xlabel(xlbl)
            if saver:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="GPU Affinity", borderaxespad=0.)
                plt.savefig(plotname, dpi=1000, bbox_inches="tight")
            if shower:
                plt.show()

def compare(dfS, dfC):
    return dfS['time']/dfc['time']

def xinterp3(dfn):
    gA = np.linspace(min(dfn.gpuA), max(dfn.gpuA), 50)
    nXA = np.linspace(min(dfn.nX), max(dfn.nX), 50)
    tpbs = np.unique(dfn.tpb)

    # There should be a combinatorial function.
    return np.array(np.meshgrid(tpbs, gA, nXA)).T.reshape(-1,3)

class perfModel(Perform):
    def __init__(self, datadf, name, typ="all", vargs={}):
        #typ needs to tell it what kind of parameters to use
        super().__init__(datadf, name)
        self.va = vargs #Dictionary of additional arguments to stats models
        self.respvar = self.cols[-1]
        self.xo = self.cols[:-1]
        self.nFrame = self.oFrame.copy()
        if typ == "all":
            self.it = list(itertools.combinations_with_replacement(self.xo, 2))

        self.ilbl = ["|".join(k) for k in self.it] 
        for n, o in zip(self.ilbl, self.it):
            self.nFrame[n] = self.oFrame[o[0]] * self.oFrame[o[1]] 
                
        self.newcols = list(self.nFrame.columns.values)
        self.newcols.remove(self.respvar)
        
        X = self.nFrame[self.newcols]
        y = self.nFrame[self.respvar]
        sm.add_constant(X)
   
        self.logY = pd.DataFrame({self.respvar: self.oFrame[self.respvar], "log"+self.respvar: np.log(self.oFrame[self.respvar])})
        self.mdl = sm.OLS(y, X)
        self.res = self.mdl.fit() 
        
    def makedf(self, lists):

        if isinstance(lists[0], list):
            dct = {k:[] for k in self.xo}
            for a in lists:
                for c, b in zip(self.xo, a):
                    dct[c].append(b)
                    
        else:
            dct = dict(zip(self.xo, lists))
            dct = {0: dct}
                
            return pd.DataFrame.from_dict(dct, orient='index')
    
        
    def transform(self, frame):     
        fr = frame.copy()
        for n, o in zip(self.ilbl, self.it):
            fr[n] = frame[o[0]] * frame[o[1]] 
            
        return fr

    def predict(self, newVals):
        
        newFrame = self.makedf(newVals)
        newFrame = newFrame[self.xo]
        fr = self.transform(newFrame)
        return self.res.predict(fr)
        

    def model(self):
        xi = xinterp3(self.oFrame)
        xi = pd.DataFrame(xi, columns=self.xo)
        xdf = self.transform(xi)
        fdf = self.res.predict(xdf)
        xi["time"] = fdf
        return xi.set_index(self.xo[0])
    
    def byFlop(self):
        dfn = self.oFrame.copy()
        newHead="Updates/us"
        dfn[newHead] = dfn["nX"]/dfn["time"]
        
        return dfn[self.cols[:2] + [newHead]]
    
    def plotFlop(self, df, f, a):
        ncols=list(df.columns.values)
        df = df.set_index(ncols[0])
        nix=list(np.unique(df.index.values))
        plt.hold(True)
        for n, cr in zip(nix, crs):
            dfn = df.xs(n)
            print(dfn)
            dfn.plot.scatter(*ncols[-2:], ax=a, label="{:1}".format(n), color=cr)
            
        a.set_title(self.title)    
        
             
    def pr(self):
        print(self.title)
        print(self.res.summary())
        
    def plotContour(self, plotpath=".", saver=True, shower=False):
        f, ax = plt.subplots(2, 2, figsize=(14,8))
        ax = ax.ravel()
        df = self.model()
        plotname = op.join(plotpath, self.title + "Contour.pdf")
        for th, a in zip(self.uniques[0], ax):
            dfn = df.xs(th).pivot(*self.cols[1:])
            self.b = dfn
            X, Y = np.meshgrid(dfn.columns.values, dfn.index.values)
            Z=dfn.values
            mxz = np.max(np.max(Z))/1.05
            mnz = np.min(np.min(Z))*1.1
            lvl = np.linspace(mnz, mxz, 10)
            a.contourf(X, Y, Z, levels=lvl)
            # a.clabel(CS, inline=1, fontsize=10)
            a.set_xscale("log")
            a.set_ylabel("gpuA")
            a.set_xlabel(xlbl)
            a.set_title(str(th))

        f.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.0)
        f.subplots_adjust(bottom=0.08, right=0.82, top=0.9)
        plt.suptitle(self.title)
        plt.colorbar()

        if saver:
            f.savefig(plotname, dpi=1000, bbox_inches="tight")
        if shower:
            plt.show()
            
        return df

def predictNew(eq, alg, args, nprocs=8):
    oldF = mostRecentResults(resultpath)
    mkstr = eq.title() + schemes[alg].title()
    oldF.columns=cols
    oldF = oldF.xs(mkstr).reset_index(drop=True)
    oldPerf = perfModel(oldF, mkstr)
    newMod = oldPerf.model()
    
    argss = args.split()
    topics = ['tpb', 'gpuA', 'nX']
    confi = []
    for t in topics:
        ix = argss.index(t)
        confi.append(float(argss[ix+1]))
        
    return oldPerf.predict(np.array(confi))

# Change the source code?  Run it here to compare to the same
# result in the old version
def compareRuns(eq, alg, args, nprocs=8): #)mdl=linear_model.LinearRegression()):
    oldF = mostRecentResults(resultpath)
    mkstr = eq.title() + schemes[alg].title()
    oldF.columns=cols
    oldF = oldF.xs(mkstr).reset_index(drop=True)
    oldPerf = perfModel(oldF, mkstr)
    newMod = oldPerf.model()

    #Run the new configuration
    expath = op.join(binpath, eq.lower())
    tp = [k for k in os.listdir(testpath) if eq.lower() in k]
    tpath = op.join(testpath, tp[0])

    outc = runMPICUDA(expath, nprocs, alg.upper(), tpath, eqopt=args)

    oc = outc.split()
    print(oc)
    i = oc.index("Averaged")
    newTime = float(oc[i+1])
    #Would be nice to get the time pipelined from the MPI program
    argss = args.split()
    topics = ['tpb', 'gpuA', 'nX']
    confi = []
    for t in topics:
        ix = argss.index(t)
        confi.append(float(ix+1))

    oldTime = newMod.predict(np.array(confi))
    print(oldTime, newTime)
    ratio = oldTime/newTime
    print(ratio)
    return ratio


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

if __name__ == "__main__":
    recentdf = mostRecentResults(resultpath)
    recentdf.columns = cols

#    rrg = linear_model.LinearRegression()
    perfs = []
    eqs = np.unique(recentdf.index.values)

    for ty in eqs:
        perfs.append(perfModel(recentdf.xs(ty), ty))
        print("------------")
        perfs[-1].pr()

#    print("RSquared values for GLMs:")
    pp = perfs[0]
    abc = pp.byFlop()
    f, a = plt.subplots(1, 1, figsize=(14,8))
    #ax = a.ravel()
    pp.plotFlop(abc, f, a)
    
#    for p in perfs:
#        p.glmBytpb(rrg)
#        #p.plotContour(plotpath=resultpath)
#        print(p.title, p.gR2)
    
    # Now compare Swept to Classic
