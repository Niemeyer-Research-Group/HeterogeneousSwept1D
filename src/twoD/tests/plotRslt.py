



import os
import os.path as op
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

fol = "testResult"
fi = os.listdir(fol)
fi = [op.join(fol, ft) for ft in fi]

def mergeDict(d1, d2):
    for k in d2.keys():
        if k in d1.keys():
            d1[k] = mergeDict(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    
    return d1

def plotTri(dfd, io):
    fx = plt.figure()
    ktime = dfd.keys()

    for i, k in enumerate(ktime):
        if i>3: break
        ax = fx.add_subplot(io[i], projection='3d')
        # X = dddf[k].columns.values
        # Y = dddf[k].index.values

        # XX, YY = np.meshgrid(X,Y)
        doif = dfd[k].unstack().reset_index()
        doif = doif.astype('float')
        doif.columns = ["X", "Y", "U"]
        ax.plot_trisurf(np.array(doif.X), np.array(doif.Y), np.array(doif.U))
        ax.set_title("t={:.3f} (s)".format(float(k)))

    return fx

def plotContf(dfd, io):
    fx = plt.figure()
    ktime = dfd.keys()

    for i, k in enumerate(ktime):
        if i>3: break
        ax = fx.add_subplot(io[i])
        dfio = dfd[k]
        #dfio.sort_index(inplace=True)
        cs = ax.contourf(dfio.values)
        ax.set_title("t={:.3f} (s)".format(float(k)))
        fx.colorbar(cs, ax=ax, shrink=0.8)

    return fx


owl = {}
for fn in fi:
    with open(fn, 'r') as thisf:
        fd = json.load(thisf)

    owl = mergeDict(owl, fd)

kf = list(fd.keys())[0]
ext = owl[kf]
ddf = {k: pd.DataFrame(v) for k,v in ext.items()}

for k, dfi in ddf.items():
    dfi.index = dfi.index.astype(float)
    dfi.columns = dfi.columns.astype(float)
    dfi.sort_index(inplace=True)
    dfi.sort_index(axis=1, inplace=True)

dddf = pd.concat(ddf, axis=1)

ktime = list(ddf.keys())
io = []
bs = 221    
krng = len(ktime) if len(ktime) < 4 else 4

for k in range(krng):
    io.append(bs+k)

plotContf(ddf, io)

plotTri(ddf, io)

plt.show()






# # dict_of_df = {k: pd.DataFrame(v) for k,v in dictionary.items()}
# df = pd.concat(dict_of_df, axis=1)




