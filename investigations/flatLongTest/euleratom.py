
# import subprocess as sp
# import shlex
# import os
# import os.path as op
# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import time
from runatom import *

rsltfolder = op.join(thispath, "eulerResult")

def parsename(t):
    return t[1].lower() + t[0]

def renameout(flist):
    dft = []
    ty = []
    for f in flist:
        ty.append([fa.split(".")[0] for fa in f.split("_")])
        dft.append(pd.read_csv(op.join(rsltfolder, f), header=0))
    
    dfto = dft[0]
    titleo = parsename(ty[0])
    dfto.rename({"time": titleo}, axis=1)
    for i in range(1, len(dft)):
        ti = parsename(ty[i])
        dfto[ti] = dft[i]["time"]

    return dfto

if __name__ == "__main__":
    root = "./bin/Euler"
    algo = ["Atomic", "Array"]

    tpb = [32*k for k in range(2, 24, 2)]
    nx  = [2**k for k in range(11, 21)]

    times = " 1e-7 .001 20 "

    for ia in algo:
        for ib in range(2):
            rt = root + ia
            for tp in tpb:
                for x in nx:
                    strnia = rt + " {:d} {:d}".format(x, tp) + times + str(ib)
                    
                    print(strnia)
                    exstr = shlex.split(strnia)
                    proc = sp.Popen(exstr)
                    sp.Popen.wait(proc)
                    time.sleep(3)

    lst = [k for k in os.listdir(rsltfolder) if k.endswith('.csv')]

    dfFinal = renameout()
    framers = quickPlot(dfFinal)


