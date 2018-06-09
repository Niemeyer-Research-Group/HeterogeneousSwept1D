

import os
import os.path as op
import sys

import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.stats

from timing_help import *

boxfigsize = (11,9)
class RawInterp(Perform):
    def interpit(self, *args):
        self.nxi = self.oFrame.groupby(self.xo[:2])
        nxlen = len(self.nxi)
        km = self.nobs//nxlen
        tFrame = self.oFrame.sort_values('nX')
        conc = []
        for i in range(km):
            ia = i*nxlen
            ib = (i+1)*nxlen
            splits = tFrame.iloc[ia:ib, :]
            mnz = np.round(np.mean(splits.nX), -3)
            splits.loc[:, 'time'] = splits.loc[:, 'time'] * mnz/splits.loc[:, 'nX']
            splits.loc[:, 'nX'] = mnz
            conc.append(splits)

        self.iFrame = pd.concat(conc)
        return self.iFrame

def plotContour(df, axi, annot):
    x = df.columns.values
    y = df.index.values
    X, Y = np.meshgrid(x,y)
    cs = axi.contourf(X, Y, df.values)
    axi.set_ylabel(annot['yl'])
    axi.set_title(annot['ti'])
    axi.set_xlabel(annot['xl'])
    return cs

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
def compareRuns(eq, alg, args, mClass, nprocs=8, dim=1): #)mdl=linear_model.LinearRegression()):
    
    bpath = obinpath if dim == 1 else tbinpath
    tpath = otestpath if dim == 1 else ttestpath
    oldF = mostRecentResults(resultpath)
    mkstr = eq.title() + schs[alg].title()
    oldF = oldF.xs(mkstr).reset_index(drop=True)
    oldPerf = mClass(oldF, mkstr)

    #Run the new configuration
    expath = op.join(bpath, eq.lower())
    tp = [k for k in os.listdir(tpath) if eq.lower() in k]
    tpath = op.join(tpath, tp[0])
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

def plotRaws(iobj, subax, respvar, nstep):
    sax = makeList(subax)
    rax = makeList(respvar)
    figC = collections.defaultdict(dict)
    for sx in sax:
        for rx in rax:
            figC[sx][rx] = iobj.plotRaw(sx, rx, nstep)

    return figC

def plotmins(df, axi, sidx, stacker=['nX', 'gpuA']):
    dff = df.stack(stacker[0])
    dfff = dff.unstack(stacker[1])
    mnplace = dfff.idxmin(axis=1)
    for a, si in zip(axi, sidx):
        a.plot(mnplace[si][0], mnplace[si][1], 'r.', markersize=20)
        
    return mnplace

def contourRaw(df, typs, tytle=None, vals='time', getfig=False):
    anno = {'ti':'10000' , 'yl': 'GPU Affinity', 'xl': 'threads per block'}
    dfCont = pd.pivot_table(df, values=vals, index='gpuA', columns=['nX', 'tpb'])
    fCont, axCont = plt.subplots(2, 2, figsize=boxfigsize)
    axc = axCont.ravel()

    subidx = dfCont.columns.get_level_values('nX').unique()
    stp = math.ceil(len(subidx)/4)
    subidx = subidx[::stp]
    for axi, nx in zip(axc, subidx):
        anno['ti'] = "GridSize: {:.0e}".format(nx) 
        cs = plotContour(dfCont[nx], axi, anno)
        fCont.colorbar(cs, ax=axi, shrink=0.8)

    if tytle: fCont.suptitle(tytle + " -- " + meas[vals]) 
    
    if getfig:
        return dfCont, fCont, axc
    
    mns = plotmins(dfCont, axc, subidx)
    formatSubplot(fCont)

    saveplot(fCont, "Performance", plotDir, "RawContour"+typs+vals)
    plt.close(fCont)
    
    return mns

#Fit DataFrame with powerfit.
def getFitRs(dfp):
    dfp.astype(float, inplace=True)
    dfc = dfp.columns
    dfa = np.log(dfp)
    dfa.index = np.log(dfa.index.values)
    spcol = pd.DataFrame()
    for s in dfc:
        print(s)
        sp, inter, r2, p, std_err = scipy.stats.linregress(np.array(dfa.index), dfa[s])
        inter = np.exp(inter)
        spcol[s] = [sp, inter, r2]
        
    spcol.index = ["A", "b", "r2"]
    return spcol



if __name__ == "__main__":

    # Flags
    titles = False
    saves = True
    finalDir = True # Final Plots for the Figure
    howFarBack = 0 # How many saves ago to look

    if not saves:
        def saveplot(f, *args):
            f = makeList(f)
            for ff in f:
                ff.show()
                
            g = input("Press Any Key: ")

    recentdf, detail = getRecentResults(howFarBack)
    recentdf.sort_index(inplace=True)
    eqs = recentdf.index.unique()
    collInst = collections.defaultdict(dict)
    collFrame = collections.defaultdict(dict)

    plotDir = "FinalResultPlots" if finalDir else "_".join([str(k) for k in [detail["System"], detail["np"], detail["date"]]]) 
    dfBESTALL = pd.DataFrame()

    for ty in eqs:
        df = recentdf.xs(ty).reset_index(drop=True) 
        opt = re.findall('[A-Z][^A-Z]*', ty)
        inst = RawInterp(df, ty)
        collInst[opt[0]][opt[1]] = inst
        collFrame[opt[0]][opt[1]] = inst.interpit()

            
    speedtypes = ["Raw", "Interpolated", "Best"] # "NoGPU"]
    dfSpeed={k: pd.DataFrame() for k in speedtypes}
    collBestI = collections.defaultdict(dict)
    collBestIG = collections.defaultdict(dict)
    collBest = collections.defaultdict(dict)
    totalgpu={}
    totaltpb={}
    respvar='time'
    tt = [(k, kk) for k in inst.uniques['tpb'] for kk in inst.uniques['gpuA']]

    fgt, axgt = plt.subplots(2, 1, sharex=True)
    fio, axio = plt.subplots(2, 2, figsize=boxfigsize)
    if titles: fio.suptitle("Best interpolated run vs observation")
    perfPath = op.join(resultpath,"Performance")

    mnCoords = pd.DataFrame()

    axdct = dict(zip(eqs, axio.ravel()))

    for ke, ie in collInst.items():
        fraw, axraw = plt.subplots(1,1)
        fspeed, axspeed = plt.subplots(1,1)
        feff, axeff = plt.subplots(1,1)
   
        for ks, iss in ie.items():
            typ = ke+ks
            tytles = typ if titles else None
            axn = axdct[ke + ks]

            ists = iss.iFrame.set_index('nX')
            iss.efficient()

            mnCoords[typ] = contourRaw(iss.iFrame, typ, tytles)

            fRawS = plotRaws(iss, 'tpb', ['time', 'efficiency'], 2)
            for rsub, it in fRawS.items():
                for rleg, itt in it.items():
                    saveplot(itt, "Performance", plotDir, "Raw"+rleg+"By"+rsub+typ)
           
            if saves:
                [plt.close(kk) for i in fRawS.values() for k in i.values() for kk in k]
            
            dfBI = iss.getBest('tpb', respvar)
            dfBIG = iss.getBest('gpuA', respvar)
            totalgpu[ke+ks] = dfBI['gpuA'].apply(pd.value_counts).fillna(0)
            totaltpb[ke+ks] = dfBIG['tpb'].apply(pd.value_counts).fillna(0)
            dfBF = getBestAll(dfBI, respvar)
            dfBF = iss.efficient(dfBF.reset_index()).set_index('nX')
            dfBF['tpb'].plot(ax=axgt[0], logx=True, label=ke+ks) 
            dfBF['gpuA'].plot(ax=axgt[1], logx=True, legend=False) 
        
            if titles:
                dfBF[respvar].plot(ax=axraw, loglog=True, label=ks, title=ke+" Best Runs")
                dfBF['efficiency'].plot(ax=axeff, logx=True, label=ks, title=ke+" Best Run Efficiency") 
            else:
                dfBF[respvar].plot(ax=axraw, loglog=True, label=ks)
                dfBF['efficiency'].plot(ax=axeff, logx=True, label=ks)

            collBestI[ke][ks] = dfBI
            collBestIG[ke][ks] = dfBIG
            collBest[ke][ks] = dfBF

            #dfSpeed["NoGPU"][ke+ks] = dfBIG['time', 0.0]/dfBF['time']
            dfBF.plot(y=respvar, ax=axn, marker="", loglog=True, legend=False)
            iss.oFrame.plot(x='nX', y=respvar, ax=axn, c='gpuA', kind='scatter',  legend=False, loglog=True)
            axn.set_title(typ)
            axn.set_xlabel(xlbl)
            axn.set_ylabel("Time per timestep (us)")
            print(typ)
            dfBESTALL[typ] = dfBF["time"]
        
        dfSpeed["Raw"][ke] = ie[schemes[0]].oFrame[respvar]/ ie[schemes[1]].oFrame[respvar]
        dfSpeed["Interpolated"][ke] =  ie[schemes[0]].iFrame[respvar]/ie[schemes[1]].iFrame[respvar]
        dfSpeed["Best"][ke] = collBest[ke][schemes[0]][respvar]/collBest[ke][schemes[1]][respvar]

        if titles:
            dfSpeed['Best'][ke].plot(ax=axspeed, logx=True, title=ke+" Speedup")
        else: 
            dfSpeed['Best'][ke].plot(ax=axspeed, logx=True)

        axraw.legend()
        axeff.legend()
        formatSubplot(fraw)
        formatSubplot(feff)
        formatSubplot(fspeed)
        axeff.set_ylabel(meas['efficiency'])
        axspeed.set_ylabel(meas["spd"])

        for ao in [axraw, axeff, axspeed]:
            ao.set_xlabel(xlbl)

        axraw.set_ylabel(meas[respvar])
        saveplot(fraw, "Performance", plotDir, "BestRun" + respvar + ke)
        saveplot(fspeed, "Performance", plotDir, "BestSpeedup" + ke)
        saveplot(feff, "Performance", plotDir, "BestRun" + "Efficiency" + ke)

    dfBESTALL.to_csv(op.join(op.join(perfPath, plotDir), 'BestTiming.csv'))
    dfbaa = getFitRs(dfBESTALL)
    sys.exit(-1)
    if titles:
        axgt[0].set_title('Best tpb')
        axgt[1].set_title('Best Affinity')    
        fgt.suptitle("Best Characteristics")  

    axgt[1].set_xlabel(xlbl)  
    
    hgo, lbo = axgt[0].get_legend_handles_labels()
    axgt[0].legend().remove()
      
    fgt.legend(hgo, lbo, 'upper right')
    saveplot(fgt, "Performance", plotDir, "BestRunCharacteristics")

    formatSubplot(fio)  
    saveplot(fio, "Performance", plotDir, "BestLineAndAllLines")
    plt.close('all')
    
    fitpb, axitpb = plt.subplots(2, 2, figsize=boxfigsize)
    figpu, axigpu = plt.subplots(2, 2, figsize=boxfigsize)
    if titles:
        fitpb.suptitle('Threads per block at best Affinity')
        figpu.suptitle('Affinity at best threads per block')

    axT = dict(zip(eqs, axitpb.ravel()))
    axG = dict(zip(eqs, axigpu.ravel()))

    for ke in collBestIG.keys():
        for ks in collBestIG[ke].keys():
            k = ke + ks
            tytles = k if titles else None
            bygpu = rowSelect(collBestIG[ke][ks], 5)
            bytpb = rowSelect(collBestI[ke][ks], 5)
            bygpu[respvar].T.plot(ax=axG[k], logy=True, title=tytles)
            bytpb[respvar].T.plot(ax=axT[k], logy=True, legend=False, title=tytles)
            hd, lb = axG[k].get_legend_handles_labels()
            axG[k].legend().remove()

    fitpb.legend(hd, lb, 'upper right')
    figpu.legend(hd, lb, 'upper right')
    formatSubplot(fitpb)
    formatSubplot(figpu)
    saveplot(figpu, "Performance", plotDir, "Besttpb vs gpu")
    saveplot(fitpb, "Performance", plotDir, "BestGPU vs tpb")

    plt.close('all')

    #VS MPI ONLY.

    # fngpu, angpu = plt.subplots(1, 1)
    # fngpu.suptitle("Speedup from GPU")
    # for k, it in dfSpeed['NoGPU'].items():
    #     it.plot(ax=angpu, logx=True, label=k)

    # angpu.legend()
    # formatSubplot(fngpu)
    # angpu.set_ylabel(meas["spdg"])
    # saveplot(fngpu, "Performance", plotDir, "HybridvsGPUonly")

    bestGpuTotal=pd.DataFrame(index=iss.iFrame['gpuA'].unique())
    bestTpbTotal=pd.DataFrame(index=iss.iFrame['gpuA'].unique())

    for k in totaltpb.keys():
        bestGpuTotal[k]=totalgpu[k].sum(axis=1)
        bestTpbTotal[k]=totaltpb[k].sum(axis=1)

    bestGpuTotal.fillna(0, inplace=True)
    bestTpbTotal.fillna(0, inplace=True)
