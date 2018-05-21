

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

owl = {}
for fn in fi:
    with open(fn, 'r') as thisf:
        fd = json.load(thisf)

    owl = mergeDict(owl, fd)

kf = list(fd.keys())[0]
ext = owl[kf]
ddf = {k: pd.DataFrame(v) for k,v in ext.items()}
dddf = pd.concat(ddf, axis=1)

ktime = ddf.keys()
io = []
if len(ktime) > 4:
    bs = 221
    io = []
    for k in range(4):
        io.append(bs+k)

fx = plt.figure()

for i, k in enumerate(ktime):
    if i>3: break
    ax = fx.add_subplot(io[i], projection='3d')
    # X = dddf[k].columns.values
    # Y = dddf[k].index.values

    # XX, YY = np.meshgrid(X,Y)
    doif = dddf[k].unstack().reset_index()
    doif = doif.astype('float')
    doif.columns = ["X", "Y", "U"]
    ax.plot_trisurf(np.array(doif.X), np.array(doif.Y), np.array(doif.U))
    ax.set_title("t={:.3f} (s)".format(float(k)))


plt.show()






# # dict_of_df = {k: pd.DataFrame(v) for k,v in dictionary.items()}
# df = pd.concat(dict_of_df, axis=1)


