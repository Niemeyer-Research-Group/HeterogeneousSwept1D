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
                
## RESULT HELP
                
