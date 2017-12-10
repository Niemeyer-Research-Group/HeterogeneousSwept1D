'''
    Python Classes and functions for running the cuda programs and
    parsing/plotting the performance data.
'''

import os
import os.path as op
import matplotlib.pyplot as plt
# from cycler import cycler
#
# import palettable.colorbrewer as pal

import pandas as pd
import shlex
import subprocess as sp
import collections
import json as j

thispath = op.abspath(op.dirname(__file__))
toppath = op.dirname(thispath)
spath = op.join(toppath, "src")
binpath = op.join(spath, "bin")
rspath = op.join(spath, "rslts")
resultpath = op.join(toppath, "results")
testpath = op.join(spath, "tests")

def numerica(df):
    df.columns = pd.to_numeric(df.columns.values)
    df.index = pd.to_numeric(df.index.values)
    df.sort_index(inplace=True)
    df = df.interpolate()
    return df.sort_index(axis=1)

def dictframes(d, t):
    if t>3:
        return {dk: dictframes(d[dk], t-1) for dk in d.keys()}
    else:
        return numerica(pd.DataFrame(d))

def depth(d, level=1):
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)

def readj(f):
    fobj = open(f, 'r')
    fr = fobj.read()
    fobj.close()
    return j.loads(fr)

# def writej(f, j):
#     fobj = open(f, 'w')
#     fr = fobj.read()
#     fobj.close()
#     return "Wrote"

def undict(d, kind='dict'):
    dp = depth(d)
    if dp>2:
        return {float(dk): undict(d[dk]) for dk in d.keys()}
    else:
        if kind=="tuple":
            return sorted([(int(k), float(v)) for k, v in d.items()])
        elif kind=="dict":
            return {int(k): float(v) for k, v in sorted(d.items())}


def makeList(v):
    if isinstance(v, collections.Iterable):
        return v
    else:
        return [v]

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

#Divisions and threads per block need to be lists (even singletons) at least.
def runMPICUDA(exece, nproc, scheme, eqfile, mpiopt="", outdir=" rslts ", eqopt=""):

    # if n[-1] != " ": n += " " for each function input.

    runnr = "mpirun -np "
    print("---------------------")
    os.chdir(spath)
    testpath = op.join(spath, "tests")

    execut = runnr + "{0} ".format(nproc) + mpiopt + exece + scheme + eqfile + outdir + eqopt

    print(execut)
    exeStr = shlex.split(execut)
    proc = sp.Popen(exeStr, stdout=sp.PIPE)
    cout, err = proc.communicate()

    return cout

# Read notes into a dataFrame. Sort by date and get sha

def mostRecentResults(rpath):
    note = readj(op.join(rpath, "notes.json"))
    hfive = op.join(rpath, "rawResults.h5")
    nframe = pd.DataFrame.from_dict(note).T
    nframe = nframe.sort_values("date")
    sha = nframe.index.values[0]
    hframe = pd.HDFStore(hfive)
    outframe = hframe[sha]
    hframe.close()
    return outframe
