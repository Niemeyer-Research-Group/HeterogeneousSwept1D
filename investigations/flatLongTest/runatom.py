
import subprocess as sp
import shlex
import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

plt.switch_backend("agg")

thispath = op.abspath(op.dirname(__file__))
impath = op.join(thispath, "images")

ext = ".pdf"
plt.style.use("swept.mplstyle")

def savePlot(fh, name):
    plotfile = op.join(impath, name + ext)
    fh.savefig(plotfile, dpi=200, bbox_inches="tight")


def plotdf(df, subp, name, kwarg={}):
	icols = df.columns.values
	dframe = {}
	fx, ax = plt.subplots(*subp, figsize=(10,8))
	axx = ax.ravel()
	ccols = list(icols[:2])
	for ix, ic in enumerate(icols[2:]):
		cidx = ccols+[ic]
		ibo = df[cidx].pivot(*cidx).T
		dframe[ic] = ibo
		ibo.plot(ax=axx[ix], title=ic, **kwarg)

	fx.subplots_adjust(bottom=0.08, right=0.85, top=0.9, wspace=0.15, hspace=0.3)
	fx.tight_layout()
	savePlot(fx, name)
	return dframe

def minkey(df, ky):
	dfo = df.groupby(ky).min(axis=1).drop('tpb', axis=1)
	return dfo

def quickPlot(f):
	if type(f) == str:
		idf = pd.read_csv(f, header=0)
		name = f.split(".")[0]
	else:
		idf = f.copy()
		name = "AtomicArrayEuler"
		

	numx = "dv" if "dv" in idf.columns.values else "nX"
	idf.rename({numx: "gridSize"}, axis=1)
	icols = idf.columns.values
	
	plotdf(idf, [2,2], name+"Raw", kwarg={"loglog": True})
	speedup = idf[icols[:2]]
	speedup[icols[2][:5]] = idf[icols[3]] / idf[icols[2]]
	speedup[icols[4][:5]] = idf[icols[5]] / idf[icols[4]]
	# speedup.columns 
	plotdf(speedup, [1, 2], name+"Speedup", kwarg={"logx": True})

	return idf, speedup

def getPlotBest(f):
	idf = pd.read_csv(f, header=0)
	minz = idf.groupby('dv').min().drop('tpb', axis=1)
	mcol = minz.columns.values
	speedz = pd.DataFrame()
	print(minz)
	speedz["Swept"] = minz[mcol[1]]/minz[mcol[0]]
	speedz["Classic"] = minz[mcol[3]]/minz[mcol[2]]
	ax = speedz.plot(logx=True)
	ax.set_ylabel("Speedup")
	ax.set_xlabel("Number of Spatial Points")
	savePlot(ax.figure, "Speedup" + f.split(".")[0][-6:])
	return speedz

if __name__ == "__main__":

	if len(sys.argv) > 1: 
		print("Just Reloading")
		sys.exit(-1)

	tpb = [2**k for k in range(5,11)]
	nx  = [2**k for k in range(11,21)]

	ex = "./bin/KSmain "

	times = " 0.001 10"

	for t in tpb:
		for n in nx:
			strnia = ex + "{:d} {:d}".format(n, t) + times

			exstr = shlex.split(strnia)
			proc = sp.Popen(exstr)
			sp.Popen.wait(proc)
			time.sleep(3)

	ss = "atomicArray.csv"
	# framed = quickPlot(ss)
	# plt.show()

