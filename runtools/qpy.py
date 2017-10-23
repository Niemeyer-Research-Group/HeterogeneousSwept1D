
import os
import os.path as op
import sys
import result_help as rh

ext = ".json"
thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
toppath = op.dirname(thispath)
spath = op.join(toppath, "src")
rspath = op.join(spath, "rslts")

f = "sEuler" + ext
sp = (2, 2)

fi = op.join(rspath, f)

jdf = rh.Solved(fi)
meta = jdf.meta
mydf = jdf.ddf
fg, axi = plt.subplots(sp[0], sp[1])
jdf.metaparse(meta)
jdf.plotResult(fg, axi)
dff = jdf.ddf
ddfk = list(dff.keys())
dsam = dff[ddfk[0]]
colss = dsam.columns.values.tolist()