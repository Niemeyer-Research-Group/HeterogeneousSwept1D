'''
    Get Loads of Results from Affinity Run
'''

import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

affpath = op.abspath(op.dirname(__file__))
resultpath = op.dirname(affpath)
toppath = op.dirname(resultpath)
pypath = op.join(toppath, "runtools")
datapath = op.join(affpath, "rslts")

sys.path.append(pypath)

from main_help import *
import timing_analysis as ta
import timing_help as th

import matplotlib.colors as mc

#SET THIS TO SAVE OR SHOW FIGS
savFig=False

hxMap=mc.ListedColormap(th.hxColors)

kodiakX=[int(k) for k in [5e6, 2e7, 4e7, 6e7]]

nTime = lambda df, x, y, z: df[x]*df[y]/df[z]

nPerf = "normTime"

ylbl = "time per timestep (us)"

ffs=np.format_float_scientific

def normalizeGroups(dfi):
    dft=dfi.copy()
    dft["grp"] = None
    for kx in kodiakX:
        dft.loc[(dfi["nX"] < kx+1e7) & (dfi["nX"] > kx-1e7), "grp"] = kx
    
    dft[nPerf] = nTime(dft, "time", "grp", "nX")
    return dft

def summaryIndiv(dfi):
    if nPerf not in dfi.columns:
        dfi = normalizeGroups(dfi)
        
    dfout=pd.DataFrame(columns=["NoGpu","Gpu"], index=dfi["grp"].unique())
    
    for k, i in dfi.groupby("grp"):
        dfout.loc[k, "NoGpu"] = i.loc[i["gpuA"] < 2, nPerf].min()    
        dfout.loc[k, "Gpu"] = i.loc[i["gpuA"] > 2, nPerf].min()
    
    dfout["Speedup"] = dfout["NoGpu"]/dfout["Gpu"]
    return dfout

def summarizeGPU(ddf, fcollect):
    dflist=[]
    for kdf, dfi in ddf.items():
        dfs = fcollect(dfi)
        dfs.columns = pd.MultiIndex.from_product([[kdf], dfs.columns])
        dflist.append(dfs)
    
    dfout = pd.concat(dflist, axis=1)
    return dfout

def headToCol(dfa):
    nms=["idx", "case"]
    d=dfa.stack(0)
    d.index.names=nms
    di=d.reset_index(level=nms[1])
    return di

def getTpbCount(dfa, typ):
    cases= ["gpuA", "grp", "case"]
    dff   = headToCol(dfa)
    dff.drop(["nX", "time"], axis=1, inplace=True)
    dff.reset_index(drop=True,inplace=True)
    kidx = dff.groupby(cases)['normTime'].idxmin()
    dffin=dff.loc[kidx]
    return dffin.groupby(typ)["tpb"].value_counts().to_frame().unstack(0)["tpb"].fillna(0).astype(int)
    

def squaredf(df, cols, deg=2):
    dfc=df.copy()
    for c in cols:
        for k in range(2,deg+1):
            nc=c+str(k)
            dfc[nc]=df[c]**k
    
    return dfc

def summarizeTPBAFF(ddf, deg=1, inter=False, xcoli=["tpb", "gpuA"]):
    idx = pd.MultiIndex.from_product([list(ddf.keys()), kodiakX])
    
    if inter: xcoli=xcoli + ["tpb-gpuA"]
        
    xcol=xcoli+[x+str(k) for x in xcoli for k in range(2,deg+1)] 
    
    dfmod = pd.DataFrame(index=idx, columns=xcol + ["const", "rsq", "rsqa"], dtype=float)

    modcoll={}
    for kdf, dfi in ddf.items():
        if inter:
            dfi["tpb-gpuA"] = dfi["tpb"]*dfi["gpuA"]
        for kx, dx in dfi.groupby("grp"):
            
            X = squaredf(dx[xcoli], xcoli, deg)
                
            X = sm.add_constant(X)
            mod=sm.OLS(dx[nPerf], X)
            res=mod.fit()
            modcoll[(kdf, kx)] = res
            rser=pd.Series({"rsq": res.rsquared, "rsqa": res.rsquared_adj})
            dfmod.loc[kdf,kx] = res.params.append(rser)
    
    return dfmod, modcoll
            
def pmods(mods, f=print):
    pr=True
    kold=list(mods.keys())[0][0]
    for k, m in mods.items():
        if pr: f("## " + k[0] + "\n\n")

        pr=False
        f("#### " + str(k[1]) + "\n\n")
        f(str(m.summary()) + "\n\n")
        if not k[0] == kold:
            pr=True
        kold=k[0]
        
def plotmdl(df, ti, axi, yf, nf):
    xv = np.arange(0,nf)
    yfx=lambda df: yf(df, xv)
    yv = df.apply(yfx, axis=1).apply(pd.Series).T

    yv["GPU Affinity"] = xv
    yv.set_index("GPU Affinity", inplace=True)     
    yv.plot(ax=axi, title=ti, markersize=0, cmap=hxMap)
    if axi.colNum == 0:
        axi.set_ylabel(ylbl)
    if axi.rowNum == 0:
        axi.set_xlabel("")
    
    axi.set_xlim([-10,210])
    handles, labels = axi.get_legend_handles_labels()
    labels=[ffs(int(k), trim="-", exp_digits=1) for k in labels]
    axi.legend(handles, labels, title="Grid Size")
    return yv
    
        
if __name__ == "__main__":
    
    pltpth = op.join(affpath, "AffinityPlots")
    dpath = datapath 
    if len(sys.argv) == 2:
        dpath = op.join(datapath, sys.argv[1])
        pltpth = op.join(op.join(affpath, "AffinityPlots"), sys.argv[1])

    os.makedirs(pltpth, exist_ok=True)
    timeFrame = readPath(dpath)
    coll = dict()
    annodict = {}

    bestCollect = pd.DataFrame(columns=list(timeFrame.keys()))
    for kType, iFrame in timeFrame.items():
        thisdf = ta.RawInterp(iFrame, kType)
        
        keepdf = thisdf.interpit()
        figt, mnT = ta.contourRaw(keepdf, kType, getfig=True)
        
        keepEfficiency = thisdf.efficient(keepdf)
        fige, _ = ta.contourRaw(keepEfficiency, kType, vals="efficiency", getfig=True, minmax="max")
        
        bestCollect[kType] = mnT
        fige.suptitle(kType+" Efficiency")
        figt.suptitle(kType+" Timing")
        plotname = op.join(pltpth, "RawContour" + kType + "Time" + ".pdf")
        plotnameeff = op.join(pltpth, "RawContour" + kType + "Efficiency" + ".pdf")
        if savFig:
            figt.savefig(plotname, bbox_inches='tight')
            fige.savefig(plotnameeff, bbox_inches='tight')
        
        coll[kType] = normalizeGroups(iFrame)
        

    dfo         = summarizeGPU(coll, summaryIndiv)
    dfall       = summarizeGPU(coll, lambda x: x.copy())
    dfm, mods   = summarizeTPBAFF(coll, 2, xcoli=["gpuA"])
    ctpbCase    = getTpbCount(dfall, "case")
    ctpbSize    = getTpbCount(dfall, "grp")  
    ctpbGpua    = getTpbCount(dfall, "gpuA")      
    ctpbGpua.columns = [int(k) for k in ctpbGpua.columns]
    
    #Paths to writeout
    sgpuPath    = op.join(dpath, "summaryGPU.csv")
    fullModels  = op.join(dpath, "GpuAvsGridDimModels.md")
    koplot      = op.join(dpath, "GPUA_model.pdf")
    kompplot    = op.join(dpath, "MethodCompare.pdf")
    speedout    = op.join(dpath, "SweepSpeedup.csv")
    scaleout    = op.join(dpath, "SweepScale.csv")
    Barout2     = op.join(dpath, "BestTpbs-nx_problem.pdf")
    Barout1     = op.join(dpath, "BestTpbs-gpuA.pdf")
    
    doff = lambda dfun, x: x**2*dfun["gpuA2"] + x*dfun["gpuA"] + dfun["const"]

    f, ax = plt.subplots(2,2, figsize=(12,10))
    f.suptitle("Kodiak hSweep Affinity Test")
    axx = ax.ravel()
    for a, k in zip(axx, coll.keys()):
        df=coll[k]
        ncolor=len(df["grp"].unique())
        df.plot.scatter(x="gpuA", y=nPerf, c="grp", cmap=hxMap,
                colorbar=False, legend=False, ax=a)
        a.set_xlabel("")
        a.set_ylabel("")

    for a, k in zip(axx, dfm.index.get_level_values(level=0).unique()):
        plotmdl(dfm.loc[k], k, a, doff, 200)
        

    sweeps=[f for f in dfo.columns.get_level_values(0).unique() if "Swept" in f]
    dfc = pd.DataFrame()
    dx="Gpu"
    fgn, ax = plt.subplots(1,2, figsize=(12,5))
    axx=ax.ravel()
    dfmove=[]
    dft=[]
    for s, a in zip(sweeps, axx):
        c=s.replace("Swept","Classic")
        dfc[s]=dfo[c][dx]/dfo[s][dx]
        dfxx=dfo.loc[:,([s,c], dx)]
        dfxx.columns=dfxx.columns.droplevel(1)
        dfxx.index.names=["Grid Size"]
        ty=s.replace("Swept","")
        dfxx.columns = [d.replace(ty, "") for d in dfxx.columns]
        dfxx.plot(ax=a, title=ty)
        if a.colNum == 0: 
            a.set_ylabel(ylbl)
        else:
            a.get_legend().remove()
        
        acc=a.get_xlim()
        fac=acc[0]*.5
        a.set_xlim([acc[0]-fac, acc[1]+fac])
        dfxx.reset_index(inplace=True)
        changemat=dfxx.iloc[1:,].values/dfxx.iloc[:-1,].values
        dfxlo=pd.DataFrame(data=changemat, columns=dfxx.columns)
        dfxlo.columns=["Delta-GridSize"] + list(dfxlo.columns[1:].values)
        dfxlo.set_index("Delta-GridSize", inplace=True)
        dfmove.append(dfxlo)
        dft.append(ty)
    
    dfscale=pd.concat(dfmove, axis=1, keys=dft)
    fog, ax = plt.subplots(1, 2, sharey=True, figsize=(12,5))
    fga, ag = plt.subplots(1, 1, figsize=(6,5))
    fog.suptitle("Frequency of best tpb")
    fga.suptitle("Frequency of best tpb by gpuA")
    axx=ax.ravel()
    topct = lambda df: df/df.sum()*100.0
    cs = topct(ctpbSize).T
    cg = topct(ctpbGpua).T
    cc = topct(ctpbCase).T
    #And there is still some axis, saving, formatting to go.
    ctpbSize.columns = [ffs(int(k), trim="-", exp_digits=1) for k in ctpbSize.columns]
    for a, c in zip(axx, (ctpbCase, ctpbSize)):
        c.plot.bar(ax=a)
        a.legend().set_title("")
        if not a.colNum:
            a.set_ylabel("Frequency of Best Outcome (%)")
    
    ctpbGpua.plot.bar(ax=ag)
    ag.set_ylabel("Frequency of Best Outcome (%)")
    fog.savefig(Barout2, bbox_inches='tight')
    fga.savefig(Barout1, bbox_inches='tight')
    
    if savFig:
        dfscale.to_csv(scaleout)
        dfc.to_csv(speedout)
        dfout.to_csv(sgpuPath)
        f.savefig(koplot, bbox_inches='tight')
        fgn.savefig(kompplot, bbox_inches='tight')
        with open(fullModels, "w") as fm:
            pmods(mods, fm.write)
    else:
        plt.show()
    
