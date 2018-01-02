'''
    Ignore this.  Using this file to save old classes and functions that may 
    still be useful.
'''

## MAIN HELP!!

#Divisions and threads per block need to be lists (even singletons) at least.
def runCUDA(Prog, divisions, threadsPerBlock, timeStep, finishTime, frequency,
    decomp, varfile='temp.dat', timefile=""):

    threadsPerBlock = makeList(threadsPerBlock)
    divisions = makeList(divisions)
    for tpb in threadsPerBlock:
        for i, dvs in enumerate(divisions):
            print("---------------------")
            print("Algorithm #divs #tpb dt endTime")
            print(decomp, dvs, tpb, timeStep, finishTime)

            execut = Prog +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(dvs, tpb, timeStep,
                    finishTime, frequency, decomp, varfile, timefile)

            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)

    return None


## TIMING HELP!!


class PerformOld(object):

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
                
                
def parseJdf(fb):
    if isinstance(fb, str):
        jdict = readj(fb)
    elif isinstance(fb, dict):
        jdict=fb

    return jdict

def plotItBar(axi, dat):
    rects = axi.patches
    for r, val in zip(rects, dat):
        axi.text(r.get_x() + r.get_width()/2, val+.5, val, ha='center', va='bottom')

    return


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
                
## RESULT HELP
                
