from timing_help import *

class PolyValued(Perform):
    def __init__(self, datadf, name):
        super().__init__(datadf, name)

    def interpit(self, respvar, bycol=['tpb', 'gpuA'], deg=2, expo="No"):
        subj='nX'
        if 'nX' not in bycol:
            xint = self.nxRange(100)
        elif 'gpuA' not in bycol:
            subj = 'gpuA'
            xint = np.around(np.linspace(self.minmaxes['gpuA'][0], self.minmaxes('gpuA'), 50), decimals=1)

        fCollect = []
        self.polys = collections.defaultdict(dict)
        dff = pd.DataFrame(index=xint)
        thisFrame = self.oFrame if self.iFrame.empty else self.iFrame

        for (k, kk), g in thisFrame.groupby(bycol):
            dff[bycol[0]]=k  
            dff[bycol[1]]=kk
            #g.loc[0] = [0, 0]
            pfit = np.poly1d(np.polyfit(g[subj], g[respvar], deg))
            self.polys[k][kk] = pfit
            dff[respvar] = pfit(xint)
            fCollect.append(dff.assign(time=pfit(xint)))
            #dff.drop(respvar, axis=1, inplace=True)
            
        self.bme = fCollect
        newpd = pd.concat(fCollect).reset_index().rename(columns={'index': subj})
        
        if self.iFrame.empty:
            self.iFrame = newpd

        return newpd


class Interp1(Perform):
    def __init__(self, datadf, name):
        super().__init__(datadf, name)

    def interpit(self, respvar, bycol=['tpb', 'gpuA']):
        subj='nX'
        if 'nX' not in bycol:
            xint = self.nxRange(100)
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
        self.mod = smf.rlm(formula=self.fm, data=self.oFrame, M=sm.robust.norms.HuberT()).fit()

    def interpit(self, respvar, bycol=['tpb', 'gpuA'], ):
        xint = self.simpleNGrid()
        it = cartProd(xint)
        self.iFrame = pd.DataFrame(it, columns=self.xo)
        self.iFrame[respvar] = self.mod.predict(self.iFrame)

        return self.iFrame

    def __formu(self):
        form = self.xo[:]
        for i, x in enumerate(self.xo):
            for xa in self.xo[i:]:
                if x == xa:
                    form.append("I(" + xa + "**2)")
                else:
                    form.append(x + ":" + xa)

        return self.cols[-1] + " ~ " + " + ".join(form)

    def pr(self):
        print(self.title)
        print(self.fm)
        print(self.mod.summary())

    def plotResid(self, saver=True):
        for x in self.xo:
            fig = sm.graphics.plot_regress_exog(self.mod, x)
            fig.suptitle(self.title + " " + x)

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

def checkInput(spec, roll):
    while spec not in roll.keys():
        print(roll.keys())
        spec = input("Those are the available classes, choose one and enter its full name: ")

    return roll[spec]

if __name__ == "__main__":

    classRoll = {'Interp1': Interp1, "StatsMods": StatsMods, "PolyValued": PolyValued}
    print(sys.argv)

    plotspec = 0
    useClass =  Interp1

    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            try:
                plotspec = int(a)
            except:
                useClass = checkInput(a, classRoll)
        
    if plotspec:
        def saveplot(f, *args):
            f = makeList(f)
            for ff in f:
                ff.canvas.draw()

    recentdf, detail = getRecentResults(0)
    eqs = recentdf.index.unique()
    collFrame = collections.defaultdict(dict)
    plotDir="_".join([str(k) for k in [detail["System"], detail["np"], detail["date"]]]) 

    for ty in eqs:
        df = recentdf.xs(ty).reset_index(drop=True) 
        opt = re.findall('[A-Z][^A-Z]*', ty)
        collFrame[opt[0]][opt[1]] = useClass(df, ty)

    speedtypes = ["Raw", "Interpolated", "Best", "NoGPU"]
    dfSpeed={k: pd.DataFrame() for k in speedtypes}
    collBestI = collections.defaultdict(dict)
    collBestIG = collections.defaultdict(dict)
    collBest = collections.defaultdict(dict)
    totalgpu={}
    totaltpb={}
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
            df = iss.interpit(respvar)

            ists = iss.iFrame.set_index('nX')
            iss.efficient()
            fRawT = iss.plotRaw('tpb', 'time', 2)
            saveplot(fRawT, "Performance", plotDir, "RawTimingTpb" + ke + ks)
            fRawE = iss.plotRaw('tpb', 'efficiency', 2)  
            saveplot(fRawE, "Performance", plotDir, "RawEfficiencyTpb" + ke + ks)
            
            if not plotspec:
                [plt.close(k) for k in fRawE + fRawT]
                
            dfBI = iss.getBest('tpb', respvar)
            dfBIG = iss.getBest('gpuA', respvar)
            totalgpu[ke+ks] = dfBI['gpuA'].apply(pd.value_counts).fillna(0)
            totaltpb[ke+ks] = dfBIG['tpb'].apply(pd.value_counts).fillna(0)
            dfBF = getBestAll(dfBI, respvar)
            dfBF = iss.efficient(dfBF.reset_index()).set_index('nX')
            dfBF['tpb'].plot(ax=axgt[0], logx=True, label=ke+ks) 
            dfBF['gpuA'].plot(ax=axgt[1], logx=True, legend=False) 

            dfBF[respvar].plot(ax=axraw, loglog=True, label=ks, title=ke+" Best Runs")
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
    saveplot(fngpu, "Performance", plotDir, "HybridvsGPUonly")

    bestGpuTotal=pd.DataFrame(index=iss.iFrame['gpuA'])
    bestTpbTotal=pd.DataFrame(index=iss.iFrame['gpuA'])

    for k in totaltpb.keys():
        bestGpuTotal[k]=totalgpu[k].sum(axis=1)
        bestTpbTotal[k]=totaltpb[k].sum(axis=1)

    bestGpuTotal.fillna(0, inplace=True)
    bestTpbTotal.fillna(0, inplace=True)