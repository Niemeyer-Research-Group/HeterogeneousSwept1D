'''
    Plots for the paper.  Takes an hd5 file.  
'''

import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

cl = ['r', 'b', 'g', 'k']
mk = ['^', 's', 'o', 'v']

ch = ['Single', 1] #Way to select what subset you want to drop.

mpl.rcParams['lines.markersize'] = 15
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams["grid.alpha"] = 0.7
mpl.rcParams["axes.grid"] = True
mpl.rcParams["figure.figsize"] = (6.46, 3.48)
mpl.rcParams["ytick.labelsize"] = "x-large"
mpl.rcParams["xtick.labelsize"] = "x-large"

fs = 22
ext = ".png"

thispath = op.abspath(op.dirname(__file__))
datapath = op.join(thispath, 'performanceData')
datafile = op.join(datapath, "performanceParsed.h5")
os.chdir(thispath)
mainpath = op.dirname(thispath)
plotpath = op.join(mainpath, 'images')

ylbl = "Time per timestep (us)"
xlbl = "Grid Size"
ylbl2 = "Speedup"
xax = np.array([2**k for k in range(11,21)])

data = pd.HDFStore(datafile)
probs = ['Euler', 'Heat', 'KS']
ylimz = [[10,5000],[.4,100],[1,500]]
ylimz = [ylimz[1]]
mpiLabels = ['ClassicCPU', 'SweptCPU', 'ClassicGPU', 'SweptGPU']
clas = 'Classic'
speedlimz = [[.5,1.1],[1,10],[1,7]]
speedlimz = [speedlimz[1]]
datafiles = data['meanValues']
datafiles = datafiles.drop(ch[0], level=ch[1])
dr = ('Double','Hybrid')

\institute{\href{mailto:mageed@oregonstate.edu}{\nolinkurl{mageed@oregonstate.edu}} and {mailto:kyle.niemeyer@oregonstate.edu}{\nolinkurl{kyle.niemeyer@oregonstate.edu}}}

for ig, prob in enumerate([probs[1]]):
    df = datafiles.loc[prob]
    pa = df.index.values.tolist()   
    if dr in pa:
        df.drop(dr, inplace=True)
        pa = df.index.values.tolist()      
        
    vals, colorz, markerz, speedval = dict(), dict(), dict(), dict()

    for i, a in enumerate(sorted(pa)):
        prec= a[0]
        alg = a[1]
        print(i, a)
        if prec not in vals.keys():
            vals[prec], colorz[prec], markerz[prec] = dict(), dict(), dict()
            
        vals[prec][alg] = df.loc[a].values
        colorz[prec][alg] = cl[i]
        markerz[prec][alg]= mk[i]
        
    plotfile = op.join(plotpath, prob+'Raw'+ext)
    for k1 in sorted(vals.keys()):
        speedval[k1] = {i : vals[k1][clas]/vals[k1][i] for i in vals[k1].keys() if not i == clas}
        for k2 in sorted(vals[k1].keys()):
            lstr = k2 #legend string
            plt.loglog(xax, vals[k1][k2], color=colorz[k1][k2], marker=markerz[k1][k2], label=lstr)
    
    plt.ylabel(ylbl, fontsize=fs)
    plt.xlabel(xlbl, fontsize=fs)
    plt.ylim(ylimz[ig])    
    plt.xlim([xax[0],xax[-1]]) 
    plt.legend(["Classic", "Swept"], loc='upper left', fontsize='x-large')
    
    plt.savefig(plotfile, bbox_inches='tight')
    plt.close()
    
    plotfile = op.join(plotpath, prob+'Speedup' + ext)
    for k1 in sorted(speedval.keys()):
        for k2 in sorted(speedval[k1].keys()):
            lstr = k2 #legend string
            plt.semilogx(xax,speedval[k1][k2], color=colorz[k1][k2], marker=markerz[k1][k2], label=lstr)
            #plt.hold(True)

    plt.ylabel(ylbl2, fontsize=fs)
    plt.xlabel(xlbl, fontsize=fs)
    plt.ylim(speedlimz[ig])    
    plt.xlim([xax[0],xax[-1]]) 
#    plt.legend(loc='upper right', fontsize='x-large')

    plt.savefig(plotfile, bbox_inches='tight')
    plt.close()
    
#Easier to put KS last to plot mpi.
newVals = vals['Double']
newVals['ClassicGPU'] = newVals['Classic']
newVals['SweptGPU']=newVals['Shared']
mpidata = pd.read_csv(op.join(datapath,'KS_MPI.csv')).values[:,1:].transpose()
pData = np.vstack((mpidata,newVals[mpiLabels[2]],newVals[mpiLabels[3]]))
rw = int(pData.shape[0])

plotfile = op.join(plotpath,"mpiRawPlot"+ext)
for i in range(rw):
    plt.loglog(xax, pData[i,:], color=cl[i], marker=mk[i])

plt.ylabel(ylbl, fontsize=fs)
plt.xlabel(xlbl, fontsize=fs)
plt.ylim([1,30000])    
plt.xlim([xax[0],xax[-1]]) 

plt.legend(mpiLabels,loc='upper left', fontsize='x-large')
plt.savefig(plotfile, bbox_inches='tight')
plt.close() 

plotfile = op.join(plotpath,"mpiSpeedupPlot"+ext)
for k in range(int(rw/2)):
    plt.semilogx(xax, pData[k]/pData[k+2], color=cl[k+2], marker=mk[k+2])
    
plt.legend(mpiLabels[2:],loc='upper left', fontsize='x-large')
plt.ylabel(ylbl2, fontsize=fs)
plt.xlabel(xlbl, fontsize=fs)
plt.xlim([xax[0],xax[-1]]) 

plt.savefig(plotfile, bbox_inches='tight')
plt.close() 

data.close()
