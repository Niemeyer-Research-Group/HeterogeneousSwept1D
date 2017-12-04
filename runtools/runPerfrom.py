'''
    Run the performance test described in the paper.  
    Will save all best runs to an hd5 file in a pandas dataframe in Results folder.
    Will also save ALL timing for the last run to appropriately named text files in Results folder.
'''

import os
import os.path as op
import sys
import pandas as pd
import numpy as np
from main_help import *
import matplotlib.pyplot as plt
import timing_help as th
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import re

thispath = op.abspath(op.dirname(__file__))
toppath = op.dirname(thispath) #Top level of git repo
resultpath = op.join(toppath,'Results')
srcpath = op.join(toppath,'src') #Binary directory
rsltpath = op.join(srcpath, 'rslts') #Top level of git repo
os.chdir(thispath)

schemes = {"C": "Classic", "S": "Swept"}

tfiles = sorted([k for k in os.listdir(rsltpath) if k.startswith('t')])

res = []
eq = []
sch = []
for tf in tfiles:
    pth = op.join(rsltpath, tf) 
    opt = re.findall('[A-Z][^A-Z]*', tf)
    eq.append(opt[0])
    sch.append(schemes[opt[1][0]])
    res.append(th.Perform(pth))



#%%
rrg = linear_model.LinearRegression()

#What we need is to take this and apply it to the jsons that exist and save them as csvs. Then we never have to do it again.  Use spyder.

od = []
ti = []
cols = ["tpb", "GPUAffinity", "nX", "time"]
for i, t in enumerate(res):
    #t.plotdict(eq[i] + " " + sch[i], plotpath=resultpath)
    tb = undict(t.oDict)
    redict = {(k0, k1, k2): [v] for k0, d1 in tb.items() for k1, d2 in d1.items() for k2, v  in d2.items()}
    odict = pd.DataFrame(redict).T.reset_index()
    odict.columns = cols
    odict = odict[(odict.nX != 0)]
    ti.append(eq[i] + sch[i])
    od.append(odict)
    
hdfpath = op.join(resultpath, "rawResults.h5")    
th.longTerm(od, ti, hdfpath)



#rg = []
#ffult = pd.DataFrame()
#for i, o in enumerate(od):
#    gA = np.arange(2*o.min()[1], o.max()[1]+1, o.min()[1])
#    nXA = np.arange(2*o.min()[2], o.max()[2]+1, o.min()[2])
#    fx = np.array(np.meshgrid(tpbA, gA, nXA)).T.reshape(-1,3)
#    ox = o[cols[:-1]]
#    oy = o[cols[-1]]
#    rrg.fit(ox, oy)
#    fy = rrg.predict(fx)
#    ffy = pd.DataFrame(np.vstack((fx.T, fy.T)).T, columns=cols)
#    ffy = ffy.set_index(cols[0]).set_index(cols[1], append=True).set_index(cols[2], append=True)
#    ffyo = ffy.unstack(cols[2])
#    ffmn = pd.DataFrame(ffyo.min().unstack(0))
#    ffmn.columns = ['time']
#    rg.append(rrg)
#    ffult[eq[i] + " " + sch[i]] = ffmn.time
#    plt.title("Best Times")
#    plt.ylabel("Time per timestep (us)")
#    plt.xlabel("Grid Size")
#    plt.legend()
    
    
    
#plt.close('all')    


#names = ['n'+str(k) for k in range(nRuns)] + ['mean', 'std']
#df_all = pd.concat([store[k] for k in store.keys()], axis=1)
#df_all = pd.concat([df_all, df_all.mean(axis=1), df_all.std(axis=1)], axis=1)
#store.close()
#saver=False, shower=True
#store2 = pd.HDFStore(finalfile)
#store2.put('runs' ,df_all)
#store2.close()