

import os
import os.path as op
import sys

import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timing_help import *

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

    def stats(self):
        return 1, 0

    def compareLike(self):
        for (k, kk), g in self.nxi:
            print(g)

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
def compareRuns(eq, alg, args, mClass, nprocs=8): #)mdl=linear_model.LinearRegression()):
    oldF = mostRecentResults(resultpath)
    mkstr = eq.title() + schs[alg].title()
    oldF = oldF.xs(mkstr).reset_index(drop=True)
    oldPerf = mClass(oldF, mkstr)

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

def plotRaws(iobj, subax, respvar, nstep):
    sax = makeList(subax)
    rax = makeList(respvar)
    figC = collections.defaultdict(dict)
    for sx in sax:
        for rx in rax:
            figC[sx][rx] = iobj.plotRaw(sx, rx, nstep)

    return figC

def plotLines(cFrame):
    dfo = cFrame.oFrame
    respvar = 'time'
    subax = 'tpb'
    saxVal = dfo[subax].unique()
    legax = 'gpuA' 
    polys = cFrame.polys
    
    ff = []
    ad = {}

    for i in range(len(saxVal)//4):
        f, ai = plt.subplots(2,2)
        ap = ai.ravel()
        ff.append(f)
        for aa, t in zip(ap, saxVal[i::2]):
            ad[t] = aa

    fb = []
    ab = {}

    for i in range(len(saxVal)//4):
        fin, ain = plt.subplots(2,2)
        ap = ain.ravel()
        fb.append(fin)
        for aa, t in zip(ap, saxVal[i::2]):
            ab[t] = aa

    for k, g in dfo.groupby(subax):
        for kk, gg in g.groupby(legax):
            pnow = expFit(polys[k][kk])
            newx = gg.copy()
            newx['TimePredict'] = pnow(newx.loc[:, 'nX'])
            newx.plot(x='nX', y='TimePredict', ax=ad[k], loglog=True, grid=True, label=kk)
            newx.plot(x='nX', y=respvar, kind='scatter', ax=ad[k], loglog=True, marker='o', label="")
            newx["Residual"] = (newx['TimePredict'] - newx[respvar])/newx[respvar]
            newx.plot(x='nX', y='Residual', kind='scatter', ax=ab[k], logx=True, marker='o', legend=False)

        ad[k].set_title(k)
        ab[k].set_title(k)
        ad[k].set_ylabel(meas[respvar])
        ad[k].set_xlabel(xlbl)

        hd, lb = ad[k].get_legend_handles_labels()
        ad[k].legend().remove()

    for fi, fbi in zip(ff, fb):
        fi = formatSubplot(fi)
        fi.legend(hd, lb, 'upper right', title=legax, fontsize="medium")
        fi.suptitle(cFrame.title + " - Raw " +  respvar +  " by " + subax)
        fbi = formatSubplot(fbi)
        fbi.suptitle(cFrame.title + " - Residuals " +  respvar +  " by " + subax)

    plt.show()

if __name__ == "__main__":
        
    plotspec = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    print(plotspec, sys.argv[1])


    if plotspec:
        def saveplot(f, *args):
            f = makeList(f)
            for ff in f:
                ff.show()
                
            g = input("Press Any Key: ")

                
    recentdf, detail = getRecentResults(0)
    eqs = recentdf.index.unique()
    collInst = collections.defaultdict(dict)
    collFrame = collections.defaultdict(dict)
    collFullPivot = collections.defaultdict(dict)

    plotDir="_".join([str(k) for k in [detail["System"], detail["np"], detail["date"]]]) 

    for ty in eqs:
        df = recentdf.xs(ty).reset_index(drop=True) 
        opt = re.findall('[A-Z][^A-Z]*', ty)
        inst = RawInterp(df, ty)
        collInst[opt[0]][opt[1]] = inst
        collFrame[opt[0]][opt[1]] = inst.interpit()
        collFullPivot[opt[0]][opt[1]] = pd.pivot_table(inst.iFrame, values='time', index='nX', columns=['tpb', 'gpuA'])

            
    speedtypes = ["Raw", "Interpolated", "Best", "NoGPU"]
    dfSpeed={k: pd.DataFrame() for k in speedtypes}
    collBestI = collections.defaultdict(dict)
    collBestIG = collections.defaultdict(dict)
    collBest = collections.defaultdict(dict)
    totalgpu={}
    totaltpb={}
    respvar='time'
    tt = [(k, kk) for k in inst.uniques['tpb'] for kk in inst.uniques['gpuA']]
    stat = pd.DataFrame(columns=eqs)

    # for ke, ie in collFrame.items():
    #     for ks, iss in ie.items():
    #         iss.interpit(respvar)
    #         ohmy = plotLines(iss)


    fgt, axgt = plt.subplots(2, 1, sharex=True)
    fio, axio = plt.subplots(2, 2)
    fio.suptitle("Best interpolated run vs observation")
    axdct = dict(zip(eqs, axio.ravel()))

    for ke, ie in collInst.items():
        fraw, axraw = plt.subplots(1,1)
        fspeed, axspeed = plt.subplots(1,1)
        feff, axeff = plt.subplots(1,1)
   
        for ks, iss in ie.items():
            typ = ke+ks
            axn = axdct[ke + ks]

            ists = iss.iFrame.set_index('nX')
            iss.efficient()
            r, _ = iss.stats()
            stat[ke+ks] = r

            fRawS = plotRaws(iss, 'tpb', ['time', 'efficiency'], 2)
            for rsub, it in fRawS.items():
                for rleg, itt in it.items():
                    saveplot(itt, "Performance", plotDir, "Raw"+rleg+"By"+rsub+typ)
           
            if not plotspec:
                [plt.close(kk) for i in fRawS.values() for k in i.values() for kk in k]
                
            dfBI = iss.getBest('tpb', respvar)
            dfBIG = iss.getBest('gpuA', respvar)
            totalgpu[ke+ks] = dfBI['gpuA'].apply(pd.value_counts).fillna(0)
            totaltpb[ke+ks] = dfBIG['tpb'].apply(pd.value_counts).fillna(0)
            dfBF = getBestAll(dfBI, respvar)
            dfBF = iss.efficient(dfBF.reset_index()).set_index('nX')
            dfBF['tpb'].plot(ax=axgt[0], logx=True, label=ke+ks) 
            dfBF['gpuA'].plot(ax=axgt[1], logx=True, legend=False) 
        

            dfBF[respvar].plot(ax=axraw, logx=True, label=ks, title=ke+" Best Runs")
            dfBF['efficiency'].plot(ax=axeff, logx=True, label=ks, title=ke+" Best Run Efficiency")           
            collBestI[ke][ks] = dfBI
            collBestIG[ke][ks] = dfBIG
            collBest[ke][ks] = dfBF

            # NO.  YOU NEED TO DO THE BEST GPUA AT EACH TPB
            dfSpeed["NoGPU"][ke+ks] = dfBIG['time', 0.0]/dfBF['time']
            dfBF.plot(y=respvar, ax=axn, loglog=True, legend=False)
            iss.oFrame.plot(x='nX', y=respvar, ax=axn, c='gpuA', kind='scatter', legend=False, loglog=True)
            axn.set_title(ke+ks)
        
        dfSpeed["Raw"][ke] = ie[schemes[0]].oFrame[respvar]/ ie[schemes[1]].oFrame[respvar]
        dfSpeed["Interpolated"][ke] =  ie[schemes[0]].iFrame[respvar]/ie[schemes[1]].iFrame[respvar]
        dfSpeed["Best"][ke] = collBest[ke][schemes[0]][respvar]/collBest[ke][schemes[1]][respvar]
        dfSpeed['Best'][ke].plot(ax=axspeed, logx=True, title=ke+" Speedup")
        axraw.legend()
        axeff.legend()
        formatSubplot(fraw)
        formatSubplot(feff)
        formatSubplot(fspeed)
        axraw.set_ylabel(meas['time'])
        axeff.set_ylabel(meas['efficiency'])
        axspeed.set_ylabel("Speedup vs Classic")


        for ao in [axraw, axeff, axspeed]:
            ao.set_xlabel(xlbl)
        axraw.set_ylabel(meas[respvar])
        saveplot(fraw, "Performance", plotDir, "BestRun" + respvar + ke)
        saveplot(fspeed, "Performance", plotDir, "BestSpeedup" + ke)
        saveplot(feff, "Performance", plotDir, "BestRun" + "Efficiency" + ke)

    axgt[0].set_title('Best tpb')
    axgt[1].set_title('Best Affinity')    
    axgt[1].set_xlabel(xlbl)  
    axspeed.set_xlabel(xlbl)
    axspeed.set_ylabel(meas["spd"])
    hgo, lbo = axgt[0].get_legend_handles_labels()
    axgt[0].legend().remove()
    fgt.suptitle("Best Characteristics")    
    fgt.legend(hgo, lbo, 'upper right')
    saveplot(fgt, "Performance", plotDir, "BestRunCharacteristics")

    fio.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.5)
    fio.subplots_adjust(top=0.9, bottom=0.08, right=0.85, hspace=0.3, wspace=0.3)    
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
            bygpu[respvar].T.plot(ax=axG[k], logy=True, title=k)
            bytpb[respvar].T.plot(ax=axT[k], logy=True, legend=False, title=k)
            hd, lb = axG[k].get_legend_handles_labels()
            axG[k].legend().remove()

    fitpb.legend(hd, lb, 'upper right')
    figpu.legend(hd, lb, 'upper right')
    formatSubplot(fitpb)
    formatSubplot(figpu)
    saveplot(figpu, "Performance", plotDir, "Besttpb vs gpu")
    saveplot(fitpb, "Performance", plotDir, "BestGPU vs tpb")

    plt.close('all')

    fngpu, angpu = plt.subplots(1, 1)
    fngpu.suptitle("Speedup from GPU")
    for k, it in dfSpeed['NoGPU'].items():
        it.plot(ax=angpu, logx=True, label=k)

    angpu.legend()
    formatSubplot(fngpu)
    angpu.set_ylabel(meas["spdg"])
    saveplot(fngpu, "Performance", plotDir, "HybridvsGPUonly")

    bestGpuTotal=pd.DataFrame(index=iss.iFrame['gpuA'].unique())
    bestTpbTotal=pd.DataFrame(index=iss.iFrame['gpuA'].unique())

    for k in totaltpb.keys():
        bestGpuTotal[k]=totalgpu[k].sum(axis=1)
        bestTpbTotal[k]=totaltpb[k].sum(axis=1)

    bestGpuTotal.fillna(0, inplace=True)
    bestTpbTotal.fillna(0, inplace=True)

    stat.index=tt