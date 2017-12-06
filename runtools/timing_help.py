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
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

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

class Perform(object):

    def __init__(self, dataset):
        self.oDict = parseJdf(dataset) #Takes either dict or path to json.
        self.dpth = depth(self.oDict)
        self.dFrame = dictframes(self.oDict, self.dpth)

    def plotdict(self, eqName, plotpath=".", saver=True, shower=False, swap=False):
        for k0 in self.oDict.keys():
            plt.figure()
            ptitle = eqName + " | tpb = " + k0
            plotname = op.join(plotpath, eqName + k0 + ".pdf")
            if swap:
                nDict = swapKeys(self.oDict[k0])
            else:
                nDict = self.oDict[k0]
            lg = sorted(nDict.keys())
            lg = lg[1::2]
            for k1 in lg:
                x = []
                y = []
                lo = [int(k) for k in nDict[k1].keys()]
                x = sorted(lo)
                for k2 in x:
                    y.append(nDict[k1][str(k2)])

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

class Perform2(object):
    def __init__(self, df, name):
        self.oFrame = df
        self.title = name
        self.cols = list(df.columns.values)
        self.tpbs = np.unique(self.oFrame.tpb)

    def glmBytpb(self, mdl):
        iframe = self.oFrame.set_index("tpb")
        self.gframes = dict()
        self.gR2 = dict()
        i=0
        minmaxes = []
        for th in self.tpbs:
            dfn = iframe.xs(th)
            mdl.fit(dfn[cols[1:-1]], dfn[cols[-1]])
            minmaxes.append([min(dfn.gpuA), min(dfn.nX), max(dfn.gpuA), max(dfn.nX)])
            xi = xinterp(dfn)
            yi = mdl.predict(xi)
            xyi = pd.DataFrame(np.vstack((xi.T, yi.T)).T, columns=cols[1:])

            self.gframes[th] = xyi.pivot(cols[1], cols[2], cols[3])
            self.gR2[th] = r2_score(dfn[cols[-1]], mdl.predict(dfn[cols[1:-1]]))


        mnmx = np.array(minmaxes)
        self.lims = np.array((mnmx[:,:2].max(axis=0), mnmx[:,-2:].min(axis=0)))
        #self.lims = np.append(b, minmaxes[-2:].min(axis=1).values)

        return mdl

    def modelGeneral(self, mdl):
        mdl.fit(self.oFrame[cols[:-1]], self.oFrame[cols[-1]])
        return mdl

    def useModel(self, mdl):
        mdl = modelGeneral(mdl)
        xInt = xinterp(self.oFrame)
        yInt = mdl.predict(xInt)
        self.pFrame = pd.DataFrame(np.vstack((xInt.T, yInt.T)).T, columns=cols)

    def transform(self):
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
            ifx = iframe.xs(ky).pivot(cols[1], cols[2], cols[3])

            ifx.plot(logx = True, logy=True, grid=True, linewidth=2, title=ptitle)
            plt.ylabel(ylbl)
            plt.xlabel(xlbl)
            if saver:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="GPU Affinity", borderaxespad=0.)
                plt.savefig(plotname, dpi=1000, bbox_inches="tight")
            if shower:
                plt.show()

    def plotContour(self, plotpath=".", saver=True, shower=False):

        f, ax = plt.subplots(2, 2, figsize=(14,8))
        ax = ax.ravel()
        plotname = op.join(plotpath, self.title + "Contour.pdf")
        for th, a in zip(self.tpbs, ax):
            df = self.gframes[th]
            X,Y = np.meshgrid(df.columns.values, df.index.values)
            Z=df.values
            mxz = np.max(np.max(Z))/1.05
            mnz = np.min(np.min(Z))*1.1
            lvl = np.linspace(mnz, mxz, 10)
            a.pcolor(X, Y, Z, levels=lvl)
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

# Change the source code?  Run it here to compare to the same
# result in the old version
def compareRuns(eq, alg, args, np=8, mdl=linear_model.LinearRegression()):
    oldF = mostRecentResults(resultpath)
    mkstr = eq.title() + schemes[alg].title()
    oldPerf = Perform(oldF.xs(mkstr), mkstr)
    newMod = oldPerf.modelGeneral(mdl)

    #Run the new configuration
    expath = op.join(binpath, eq.lower())
    tp = [k for k in op.listdir(testpath) if eq.lower() in k]
    tpath = op.join(testpath, tp[0])

    outc = runMPICUDA(expath, np, alg.upper(), tpath, eqopt=args)

    oc = outc.split()
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

    rrg = linear_model.LinearRegression()
    perfs = []
    eqs = np.unique(recentdf.index.values)

    for ty in eqs:
        perfs.append(Perform2(recentdf.xs(ty), ty))

    print("RSquared values for GLMs:")
    for p in perfs:
        p.glmBytpb(rrg)
        #p.plotContour(plotpath=resultpath)
        print(p.title, p.gR2)

    # Now compare Swept to Classic



#def plotContour(self, plotpath=".", saver=True, shower=False):

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
