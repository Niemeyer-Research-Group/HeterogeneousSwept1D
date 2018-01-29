'''
    Python Classes and functions for running the cuda programs and
    parsing/plotting the performance data.
'''

import os
import os.path as op
import matplotlib.pyplot as plt

import pandas as pd
import shlex
import subprocess as sp
import collections
import json as j
import git
import re
from datetime import datetime

thispath = op.abspath(op.dirname(__file__))
toppath = op.dirname(thispath)
spath = op.join(toppath, "src")
onepath = op.join(spath, "oneD")
twopath = op.join(spath, "twoD")
binpath = op.join(spath, "bin")
orspath = op.join(onepath, "rslts")
trspath = op.join(twopath, "rslts")
resultpath = op.join(toppath, "results")
otestpath = op.join(onepath, "tests")
ttestpath = op.join(twopath, "tests")

schemes = {"C": "Classic", "S": "Swept"}

todaytoday = str(datetime.date(datetime.today()).isoformat().replace("-", "_"))

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
    if isinstance(v, collections.Iterable) and not isinstance(v, str):
        return v
    else:
        return [v]

#Category: i.e Performance, RunDetail (comp_nprocs_date), plottitle
def saveplot(f, cat, rundetail, titler):
    #Category: i.e regression, Equation: i.e. EulerClassic , plot
    tplotpath = op.join(resultpath, cat)
    if not op.isdir(tplotpath):
        os.mkdir(tplotpath)

    plotpath = op.join(tplotpath, rundetail)
    if not op.isdir(plotpath):
        os.mkdir(plotpath)

    if isinstance(f, collections.Iterable):
        for i, fnow in enumerate(f):
            plotname = op.join(plotpath, titler + str(i) + ".pdf")
            fnow.savefig(plotname, bbox_inches='tight')

    else:
        plotname = op.join(plotpath, titler + ".pdf")
        f.savefig(plotname, bbox_inches='tight')


#Divisions and threads per block need to be lists (even singletons) at least.
def runMPICUDA(exece, nproc, scheme, eqfile, mpiopt="", outdir=" rslts ", eqopt=""):

    runnr = "mpirun -np "
    print("---------------------")
    os.chdir(spath)
    
    execut = runnr + "{0} ".format(nproc) + mpiopt + exece + scheme + eqfile + outdir + eqopt

    print(execut)
    exeStr = shlex.split(execut)
    proc = sp.Popen(exeStr, stdout=sp.PIPE)
    ce, er = proc.communicate()

    ce = ce.decode('utf8') if ce else "None"
    er = er.decode('utf8') if er else "None"

    print(er)

    return ce

# Read notes into a dataFrame. Sort by date and get sha

def getRecentResults(nBack, prints=None):
    rpath = resultpath
    note = readj(op.join(rpath, "notes.json"))
    hfive = op.join(rpath, "rawResults.h5")
    nframe = pd.DataFrame.from_dict(note).T
    nframe = nframe.sort_values("date", ascending=False)
    sha = nframe.index.values[nBack]
    hframe = pd.HDFStore(hfive)
    outframe = hframe[sha]
    hframe.close()
    if prints:
       pr = makeList(prints)
       for ky, it in note:
           print("SHA: ", ky)
           for p in pr:
                if pr in it.keys():
                    print(pr, it[pr])
                else:
                    print(pr, " Is not a key")

    return outframe, note[sha]

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(op.join(root, name))
    return result

def swapKeys(d):
    b = collections.defaultdict(dict)
    for k0 in d.keys():
        for k1 in d[k0].keys():
            if d[k0][k1]:
                b[k1][k0] = d[k0][k1]

    return b

def parseCsv(fb):
    if isinstance(fb, str):
        jframe = pd.read_csv(fb)
    elif isinstance(fb, dftype):
        jframe = fb

    jframe = jframe[(jframe.nX !=0)]

    return jframe

def readPath(fpth):

    tfiles = sorted([k for k in os.listdir(fpth) if k.startswith('t')])

    res = []
    ti = []
    for tf in tfiles:
        pth = op.join(fpth, tf) 
        opt = re.findall('[A-Z][^A-Z]*', tf)
        ti.append(opt[0]+schemes[opt[1][0]])
        res.append(parseCsv(pth))

    return dict(zip(ti, res))

# Takes list of dfs? title of each df, longterm hd5, option to overwrite
# incase you write wrong.  Use carefully!
def longTerm(dfs, titles, fhdf, overwrite=False):
    today = todaytoday
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    nList = []
    for d, t in zip(dfs, titles):
        d["eqs"] = [t]*len(d)
        nList.append(d.set_index("eqs"))

    dfcat = pd.concat(nList)

    opStore = pd.HDFStore(fhdf)
    fnote = op.join(op.dirname(fhdf), "notes.json")
    if op.isfile(fnote):
        dnote = readj(fnote)
    else:
        dnote = dict()

    if sha in dnote.keys() and not overwrite:
        opStore.close()
        print("You need a new commit before you save this again")
        return "Error: would overwrite previous entry"

    dnote[sha] = {"date": today}
    dnote[sha]["System"] = input("What machine did you run it on? ")
    dnote[sha]["np"] = int(input("Input # MPI procs? "))
    dnote[sha]["note"] = input("Write a note for this data save: ")

    with open(fnote, "w") as fj:
        json.dump(dnote, fj)

    opStore[sha] = dfcat
    opStore.close()
    return dfcat

if __name__ == "__main__":
    #1D
    getpath = orspath

    if len(sys.argv) > 2:
        if not op.exists(op.abspath(sys.argv[1])):
            print("The path you entered doesn't exist. Make sure you enter the relative path to the result files from cwd with a period in front. Ex: ./src/rslts")
            sys.exit(1)
        else:
            getpath = op.abspath(sys.argv[1])

    rs, ty = readPath(getpath)
    hdfpath = op.join(resultpath, "rawResults.h5")    
    longTerm(rs, ty, hdfpath)
