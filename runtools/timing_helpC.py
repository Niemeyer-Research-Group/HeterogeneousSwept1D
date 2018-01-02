'''
    Classes and functions for plotting the actual results of the simulations.

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
import palettable.colorbrewer as pal
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from cycler import cycler
import random

cti = CloughTocher2DInterpolator

import result_help as rh
from main_help import *

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
#mpl.rcParams['lines.markersize'] = 8
mpl.rcParams["figure.figsize"] = 14,8
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True
mpl.rcParams["savefig.dpi"] = 1000
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.subplot.bottom"] = 0.08
mpl.rcParams["figure.subplot.right"] = 0.85
mpl.rcParams["figure.subplot.top"] = 0.9
mpl.rcParams["figure.subplot.wspace"] = 0.15
mpl.rcParams["figure.subplot.hspace"] = 0.25
mpl.rcParams["figure.subplot.left"] = 0.05

ylbl = "Time per timestep (us)"
xlbl = "Grid Size"
ext = ".json"

dftype=pd.core.frame.DataFrame

schemes = {"C": "Classic", "S": "Swept"}

cols = ["tpb", "gpuA", "nX", "time"]

crs=["r", "b", "k", "g"]

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

class Perform(object):
    def __init__(self, df, name=""):
        self.oFrame = df
        self.title = name
        self.cols = list(df.columns.values)
        self.xo = self.cols[:3]
        self.uniques, self.minmaxes = [], []
            
        for k in self.xo:
            self.uniques.append(list(np.unique(self.oFrame[k])))
            self.minmaxes.append( (min(self.uniques[-1]), max(self.uniques[-1])) ) 
            
        ap = []
        for k, g in self.oFrame.groupby('tpb'):
            for kk, gg in g.groupby('gpuA'):
                ap.append(gg['nX'].max())
                
        maxx = min(ap)
        
        ap = []
        for k, g in self.oFrame.groupby('tpb'):
            for kk, gg in g.groupby('gpuA'):
                ap.append(gg['nX'].min())
                
        minx = max(ap)
        self.minmaxes[-1] = (minx, maxx)
        
    def fullTest(self):
        nxs = np.logspace(*np.log2(self.minmaxes[-1]), base=2, num=1000)
        nxs = nxs[100:900]
#        tpbs = np.arange(self.minmaxes[0][0], self.minmaxes[0][1]+1, 32)
        gpuas = np.arange(self.minmaxes[1][0], self.minmaxes[1][1]+.01, .5)
        iters = itertools.product(gpuas, nxs)
        df = pd.DataFrame(list(iters), columns=self.xo[1:3])
#        grd = np.meshgrid(tpbs, gpuas, nxs)
        return df
            
    def saveplot(self, cat, pt):
        #Category: i.e regression, Equation: i.e. EulerClassic , plot
        tplotpath = op.join(resultpath, cat)
        if not op.isdir(tplotpath):
            os.mkdir(tplotpath)
            
        plotpath = op.join(tplotpath, self.title)
        if not op.isdir(plotpath):
            os.mkdir(plotpath)
            
        plotname = op.join(plotpath, str(pt) + self.title + ".pdf")
        plt.savefig(plotname)
        
    def __str__(self):
        ms = "%s \n %s \n Unique Exog: \n" % (self.title, self.oFrame.head())
    
        for i, s in enumerate(self.uniques):
            ms = ms + self.cols[i] + ": \n " + str(s) + "\n"
        return ms
        

class perfModel(Perform):
    def __init__(self, datadf, name, respvar):
        #typ needs to tell it what kind of parameters to use
        super().__init__(datadf, name)
        self.respvar = respvar
        
        self.xof = self.xo[:]

        for i, x in enumerate(self.xo):
            for xa in self.xo[i:]:
                if x == xa:
                    self.xof.append("I(" + xa + "**2)")
                else:
                    self.xof.append(x + ":" + xa)
        
        self.formu = self.respvar + " ~ " + " + ".join(self.xof)
            
        print(self.formu)
        self.mdl = smf.ols(formula=self.formu,  data=self.oFrame)
        self.res = self.mdl.fit() 
        self.pv = self.res.params
        
    def plotResid(self, saver=True):
        for x in self.xo:
            fig = sm.graphics.plot_regress_exog(self.res, x)
            fig.suptitle(self.title + " " + x)
            if saver:
                self.saveplot("Residual", x)
            else:
                plt.show()
     
    def plotRaw(self):
        for k, g in self.oFrame.groupby(self.xo[0]):
            if (k/32 %2):
                f, a = plt.subplots(1, 1)
                
                for kk, gg in g.groupby(self.xo[1]):
                    
                    a.semilogx(gg[self.xo[-1]], gg[self.respvar], label=str(kk))
            
                h, l = a.get_legend_handles_labels()
                plt.legend(h,l)
                plt.title(self.title + str(k))
            
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

    def predict(self, newVals):
        if not isinstance(newVals, dftype):
            newVals = self.makedf(newVals)
        
        newFrame = newVals[self.xo]
        return self.res.predict(newFrame)    
    
    def model(self):
        xi = self.fullTest()
        fdf = self.res.predict(xi)
        xi[self.repvar] = fdf
        return xi.set_index(self.xo[0])

    def bestG(self, n, th):
        return -(self.pv["gpuA"] + self.pv["tpb:gpuA"]*th + self.pv["gpuA:nX"]*n)/(2.0*self.pv["I(gpuA ** 2)"]) 
    
    def buildGPU(self):
        nxs = np.logspace(*np.log2(self.minmaxes[-1]), base=2, num=1000)
        tpbs = np.arange(self.minmaxes[0][0], self.minmaxes[0][1]+1, 32)
        tmpdf = pd.DataFrame(columns=self.xo[:-1])
        dfg = tmpdf.copy()
        tmpdf['tpb'] = tbps # self.uniques[0]
        
        for n in nxs:
            tmpdf['nX'] = n
            tmpdf['gpuA'] = self.bestG(tmpdf['nX'], tmpdf['tpb'])
            tmpdf[self.respvar] = self.predict(tmpdf)
            dfg = dfg.append(tmpdf.loc[tmpdf[self.respvar].idxmin()], ignore_index=True)
    
        return dfg

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
        print(self.res.summary(title=self.title))
        
    def plotLine(self, plotpath=".", saver=True, shower=False):
        f, ax = plt.subplots(2, 2, figsize=(14,8))
        ax = ax.ravel()
        df = self.model()
        plotname = op.join(plotpath, self.title + "Contour.pdf")
        for th, a in zip(np.unique(df.index.values), ax):
            dfn = df.xs(th).pivot(*self.cols[1:]).T
            dfn.plot(ax=a, logx=True, logy=True)
            a.set_xscale("log")
            a.set_ylabel("gpuA")
            a.set_xlabel(xlbl)
            a.set_title(str(th))

        f.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.0)
        f.subplots_adjust(bottom=0.08, right=0.82, top=0.9)
        plt.suptitle(self.title)

        if saver:
            f.savefig(plotname, dpi=1000, bbox_inches="tight")
        if shower:
            plt.show()
            
        return df
        
    def plotContour(self, plotpath=".", saver=True, shower=False):
        f, ax = plt.subplots(2, 2, figsize=(14,8))
        ax = ax.ravel()
        df = self.model()
        plotname = op.join(plotpath, self.title + "Contour.pdf")
        for th, a in zip(np.unique(df.index.values), ax):
            dfn = df.xs(th).pivot(*self.cols[1:])
            self.b = dfn
            X, Y = np.meshgrid(dfn.columns.values, dfn.index.values)
            Z=dfn.values
            mxz = np.max(np.max(Z))/1.05
            mnz = np.min(np.min(Z))*1.1
            lvl = np.linspace(mnz, mxz, 10)
            CS = a.contourf(X, Y, Z, levels=lvl)
            a.clabel(CS, inline=1, fontsize=10)
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
    
    def polyAffinity(self, tpb, noX):
        print(tpb, noX)
        CC = self.pv["Intercept"] + self.pv["tpb"]*tpb + self.pv["nX"]* noX
        CC += self.pv["tpb:nX"] * noX * tpb + self.pv["I(tpb ** 2)"] * tpb**2 + self.pv["I(nX ** 2)"] * noX**2
        b = self.pv["gpuA"] + self.pv["tpb:gpuA"]*tpb
        a = self.pv["I(gpuA ** 2)"]
        return lambda g: a * g**2 + b * g + CC
        
    
    def plotAffinity(self, ax, tpb):
        nxs = np.round(np.arange(*np.log2(self.minmaxes[-1])))[-5:]
        gr = np.linspace(*self.minmaxes[1], 100)
        nxs = np.power(2, nxs)
        for nx in nxs:
            ply = self.polyAffinity(tpb, nx)
            tm = ply(gr)
            bflop = nx/tm
            ax.plot(gr, bflop, label=str(int(nx)))
            
        h, l = ax.get_legend_handles_labels()
        ax.set(xlabel="GPU Affinity", ylabel="Grid pts/us", title="tpb | {:d}".format(int(tpb)))
        return h,l
#        
#        return gf, pG(gf)

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


if __name__ == "__main__":
    recentdf = mostRecentResults(resultpath)
    cols = list(recentdf.columns.values)
    
#    rrg = linear_model.LinearRegression()
    timed = "time"
    timeLabel = "us per timestep"
    pointSpeed = "Speed"
    pointLabel = "MGridPts/s"
    recentdf[pointSpeed] = recentdf[cols[-2]]/recentdf[cols[-1]]
    rVar = timed
    rLabel = timeLabel
    perfs = []
    eqs = list(set(recentdf.index.values))
    tpbs = set(recentdf.tpb)
    pex = Perform(recentdf.xs(eqs[0]))
    parms = pex.fullTest()    

    #nxs = parms['nX'].unique()
    finalFrame = pd.DataFrame(index=nxs, columns=eqs)
    sFrame=pd.DataFrame()
    gFrame=pd.DataFrame()
    # Flatten the parms for interpolant requiring grid.
    
    for ty in eqs:
        dfo = recentdf.xs(ty).set_index('tpb')
        f, a = plt.subplots(1, 1)
        plt.suptitle(ty)
        
        for tp in tpbs:
            df = dfo.xs(tp)
            dcol = list(df.columns.values)
            ctiFill = cti(df[dcol[:2]], df[rVar])
            tmpdf = pd.DataFrame(parms, columns=dcol[:2])
            tmpdf[rVar] = ctiFill(parms)
            sFrame[tp] = tmpdf.pivot(index='nX', columns='gpuA', values=rVar).min(axis=1)

        sFrame.plot(ax=a, logx=True)


#    
#       def plotRaw(self):
#        for k, g in self.oFrame.groupby(self.xo[0]):
#            if (k/32 %2):
#                f, a = plt.subplots(1, 1)
#                
#                for kk, gg in g.groupby(self.xo[1]):
#                    
#                    a.semilogx(gg[self.xo[-1]], gg[self.respvar], label=str(kk))
#            
#                h, l = a.get_legend_handles_labels()
#                plt.legend(h,l)
#                plt.title(self.title + str(k))
            
    
#    abc = pp.byFlop()
#    coef = ["I(tpb ** 2)", "I(gpuA ** 2)", "tpb:gpuA"] #inflection test
#    for p in perfs:
#        dft = p.res.params
#        a = 4*dft[coef[0]]*dft[coef[1]] - dft[coef[2]]**2
#        f, a = plt.subplots(2,2)
#        ax = a.ravel()
#        tpbs = p.uniques[0]
#        for i in range(4):
#            idx = int(stride * i)
#            ha, lb = p.plotAffinity(ax[i], tpbs[idx])
#
#        plt.legend(ha, lb, bbox_to_anchor=(1.05, 2), loc=2, title="Grid Size", borderaxespad=0.)
#        f.suptitle(p.title, x=0.45, fontweight='bold')
#        
#        p.saveplot("Affinity", "wkstn")
        
    
        
        
#    for p in perfs:
#        p.plotResid()
        

    

