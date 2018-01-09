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
from scipy import interpolate
from cycler import cycler
import re

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
        f.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.0)
        f.subplots_adjust(top=0.9, bottom=0.08, right=0.85)
        
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
    def __init__(self, df, name):
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
        
        drops = self.uniques[legax][1::legstep] if legstep > 1 else []
        saxVal = self.uniques[subax]
        ff = []
        ad = {}
        for i in range(len(saxVal)//4):
            f, ai = plt.subplots(2,2)
            ap = ai.ravel()
            ff.append(f)
            for aa, t in zip(ap, saxVal[i::2]):
                ad[t] = aa

            plt.suptitle(self.title + " - Raw time by " + subax)

        for k, g in self.oFrame.groupby(subax):
            gg = g.pivot(index='nX', columns=legax, values=respvar)
            gg = gg.drop(drops, axis=1)
            gg.plot(ax=ad[k], logy=True, logx=True, grid=True)
            ad[k].set_title(k)
            ad[k].set_ylabel(meas[respvar])
            ad[k].set_xlabel(xlbl)
            hd, lb = ad[k].get_legend_handles_labels()
            ad[k].legend().remove()

        for i, fi in enumerate(ff):
            fi = formatSubplot(fi)
            fi.legend(hd, lb, 'upper right', title="Threads per block", fontsize="medium")

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

    def nxRange(self, n):
        return np.logspace(*(np.log2(self.minmaxes['nX'])), base=2, num=n)

    def simpleNGrid(self):
        xint = [self.uniques['tpb'], self.uniques['gpuA']]
        xint.append(self.nxRange(100))
        return xint


class Interp1(Perform):
    def __init__(self, datadf, name):
        super().__init__(datadf, name)

    def interpit(self, bycol=['tpb', 'gpuA']):
        subj='nX'
        if 'nX' not in bycol:
            xint = np.around(self.nxRange(100), decimals=0)
        elif 'gpuA' not in bycol:
            subj = 'gpuA'
            xint = np.around(np.linspace(self.minmaxes['gpuA'][0], self.minmaxes('gpuA'), 50), decimals=1)

        fCollect = []
        dff = pd.DataFrame(index=xint)
        thisFrame = self.oFrame if self.iFrame.empty else self.iFrame
        for (k, kk), g in thisFrame.groupby(bycol):
            dff[bycol[0]]=k  
            dff[bycol[1]]=kk
            g = g.drop_duplicates(subj)
            interper = interpolate.interp1d(g[subj], g[respvar], kind='cubic')
            fCollect.append(dff.assign(time=interper(xint)))

        newpd = pd.concat(fCollect).reset_index().rename(columns={'index': subj})
        
        if self.iFrame.empty:
            self.iFrame = newpd

        return newpd


class StatsMods(Perform):
    def __init__(self, datadf, name):
        super().__init__(datadf, name)
        self.fm = self.__formu()
        self.mod = smf.ols(formula=fm, data=self.oFrame().fit())

    def interpit(self):
        xint = self.simpleNGrid()
        it = cartprod(xint)
        self.iFrame = pd.DataFrame(it, columns=self.xo)
        self.iFrame[respvar] = mod.predict(self.iFrame)

    def __formu(self):
        form = self.xo[:]
        for i, x in enumerate(xo):
            for xa in xo[i:]:
                if x == xa:
                    form.append("I(" + xa + "**2)")
                else:
                    form.append(x + ":" + xa)

        return self.cols[-1] + " ~ " + " + ".join(form)

    def pr(self):
        print(self.title)
        print(self.fm)
        print(self.smod.summary())

#------------------------------

def predictNew(eq, alg, args, nprocs=8):
    oldF = mostRecentResults(resultpath)
    mkstr = eq.title() + schs[alg].title()
    oldF.columns=cols
    oldF = oldF.xs(mkstr).reset_index(drop=True)
    oldPerf = Perform(oldF, mkstr)
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
    mkstr = eq.title() + schs[alg].title()
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
    recentdf, detail = getRecentResults(0)
    eqs = recentdf.index.unique()
    collFrame = collections.defaultdict(dict)
    plotDir="_".join([str(k) for k in [detail["System"], detail["np"], detail["date"]]]) 

    for ty in eqs:
        df = recentdf.xs(ty).reset_index(drop=True) 
        opt = re.findall('[A-Z][^A-Z]*', ty)
        collFrame[opt[0]][opt[1]] = Interp1(df, ty)

    speedtypes = ["Raw", "Interpolated", "Best", "NoGPU"]
    dfSpeed={k: pd.DataFrame() for k in speedtypes}
    collBestI = collections.defaultdict(dict)
    collBestIG = collections.defaultdict(dict)
    collBest = collections.defaultdict(dict)
    respvar='time'

    fgt, axgt = plt.subplots(2, 1, sharex=True)
    fio, axio = plt.subplots(2, 2)
    fio.suptitle("Best interpolated run vs observation")
    axdct = dict(zip(eqs, axio.ravel()))
    for ke, ie in collFrame.items():
        fraw, axraw = plt.subplots(1,1)
        fspeed, axspeed = plt.subplots(1,1)
        feff, axeff = plt.subplots(1,1)
   
        for ks, iss in ie.items():
            axn = axdct[ke + ks]
            df = iss.interpit()
            iss.efficient()
            fRawT = iss.plotRaw('tpb', 'time')
            saveplot(fRawT, "Performance", plotDir, "RawTimingTpb" + ke + ks)
            fRawE = iss.plotRaw('tpb', 'efficiency', 2)
            saveplot(fRawE, "Performance", plotDir, "RawEfficiencyTpb" + ke + ks)
            [plt.close(k) for k in fRawE + fRawT]
            dfBI = iss.getBest('tpb', respvar)
            dfBIG = iss.getBest('gpuA', respvar)
            dfBF = getBestAll(dfBI, respvar)
            dfBF = iss.efficient(dfBF.reset_index()).set_index('nX')
            dfBF['tpb'].plot(ax=axgt[0], logx=True, label=ke+ks) 
            dfBF['gpuA'].plot(ax=axgt[1], legend=False, logx=True) 

            dfBF[respvar].plot(ax=axraw, loglog=True, label=ks, title=ke+" Best Runs")
            dfBF['efficiency'].plot(ax=axeff, loglog=True, label=ks, title=ke+" Best Run Efficiency")           
            collBestI[ke][ks] = dfBI
            collBestIG[ke][ks] = dfBIG
            collBest[ke][ks] = dfBF
            dfSpeed["NoGPU"][ke+ks] = dfBF['time']/iss.iFrame.loc[iss.iFrame['gpuA']<0.1]['time']
            dfBF.plot(y=respvar, ax=axn, logx=True, legend=False)
            iss.oFrame.plot(x='nX', y=respvar, ax=axn, c='gpuA', kind='scatter', legend=False, logx=True)
            axn.set_title(ke+ks)
        
        dfSpeed["Raw"][ke] = ie[schemes[0]].oFrame[respvar]/ ie[schemes[1]].oFrame[respvar]
        dfSpeed["Interpolated"][ke] =  ie[schemes[0]].iFrame[respvar]/ie[schemes[1]].iFrame[respvar]
        dfSpeed["Best"][ke] = collBest[ke][schemes[0]][respvar]/collBest[ke][schemes[1]][respvar]
        dfSpeed['Best'][ke].plot(ax=axspeed, logy=True, logx=True, title=ke+" Speedup")
        axraw.legend()
        axeff.legend()
        saveplot(fraw, "Performance", plotDir, "BestRun" + respvar + ke)
        saveplot(fspeed, "Performance", plotDir, "BestSpeedup" + ke)
        saveplot(feff, "Performance", plotDir, "BestRun" + "Efficiency" + ke)

    axgt[0].set_title('Best tpb')
    axgt[1].set_title('Best Affinity')    
    axgt[1].set_xlabel(xlbl)  
    hgo, lbo = axgt[0].get_legend_handles_labels()
    axgt[0].legend().remove()
    fgt.suptitle("Best Characteristics")    
    fgt.legend(hgo, lbo, 'upper right')
    saveplot(fgt, "Performance", plotDir, "BestRunCharacteristics")
    formatSubplot(fio)
    saveplot(fio, "Performance", plotDir, "BestLineAndAllLines")

    plt.close('all')
    
    fitpb, axitpb = plt.subplots(2, 2)
    figpu, axigpu = plt.subplots(2, 2)
    fitpb.suptitle('Threads per block at best Affinity')
    figpu.suptitle('Affinity at best threads per block')
    axT = dict(zip(eqs, axitpb.ravel()))
    axG = dict(zip(eqs, axigpu.ravel()))

    for ke in collBestIG.keys():
        for ks in collBestIG[ke].keys():
            k = ke + ks
            bygpu = rowSelect(collBestIG[ke][ks], 5)
            bytpb = rowSelect(collBestI[ke][ks], 5)
            bygpu[respvar].T.plot(ax=axG[k],  logy=True, title=k)
            bytpb[respvar].T.plot(ax=axT[k], logy=True, legend=False, title=k)
            hd, lb = axG[k].get_legend_handles_labels()
            axG[k].legend().remove()

    fitpb.legend(hd, lb, 'upper right')
    figpu.legend(hd, lb, 'upper right')
    formatSubplot(fitpb)
    formatSubplot(figpu)
    saveplot(figpu, "Performance", plotDir, "Besttpb vs gpu")
    saveplot(fitpb, "Performance", plotDir, "BestGPU vs tpb")



