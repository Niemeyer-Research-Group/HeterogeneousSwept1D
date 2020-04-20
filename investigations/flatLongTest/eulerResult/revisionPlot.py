# @Author: Anthony Walker <sokato>
# @Date:   2020-04-17T17:31:11-07:00
# @Email:  walkanth@oregonstate.edu
# @Filename: revisionPlot.py
# @Last modified by:   sokato
# @Last modified time: 2020-04-20T14:23:16-07:00


import os
import matplotlib.pyplot as plt
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
save_path = os.path.join(curr_path,"figs")

def bestRunsSpeedUp(speedup=True):
    curr_file = os.path.join(curr_path,'bestRuns.csv')
    with open(curr_file,'r') as f:
        line = f.readline()
        line = f.readline()
        data = []
        while line:
            data.append([float(item) for item in line.split(',')])
            line = f.readline()
        f.close()
    #Normalizing data
    norm_index = 0
    data = list(zip(*data))
    ndata = np.asarray(data.pop(norm_index+1))
    data = np.asarray(data)
    size = data[0,:]
    data = data[1:,:]
    data  = ndata/data if speedup else data/ndata
    #Plot data
    fig = plt.figure()
    plt.grid(b=True)
    plt.xlabel('Grid size')
    sstr = "Speedup" if speedup else "Slowdown"
    plt.ylabel(sstr)
    # plt.xlim([0,4])
    lstyle = ['solid','dashed','dotted']
    strs = ['Classic lengthening', 'Swept flattening', 'Swept lengthening']
    colors = ['red','green','blue']
    for i in range(data.shape[0]):
        plt.semilogx(size,data[i,:],color=colors[i-1],linestyle=lstyle[i-1],linewidth=3)
        idx = len(size)//2
        plt.text(size[idx], data[i,-1], strs[i-1], fontsize=12)
    plt.savefig(os.path.join(save_path,"Best-Runs-"+sstr+".pdf"))

def fileDisect(f):
    data =[]
    line = f.readline()
    line = f.readline()[:-1]
    while line:
        lineData = [float(l) if "." in l else int(l) for l in line.split(',')]
        data.append(lineData)
        line = f.readline()[:-1]
    return data

def generateFigure(size,data,value,speedup):
    fig = plt.figure()
    sstr = "Speedup-{}".format(value) if speedup else "Slowdown-{}".format(value)
    plt.grid(b=True)
    plt.xlabel('Grid size')
    plt.ylabel(sstr)
    lstyle = ['solid','dashed','dotted']
    strs = ['Classic lengthening', 'Swept flattening', 'Swept lengthening']
    colors = ['red','green','blue']
    for i in range(data.shape[1]):
        plt.semilogx(size,data[:,i],color=colors[i-1],linestyle=lstyle[i-1],linewidth=3)
        idx = len(size)//2
        plt.text(size[idx], data[-1,i], strs[i-1], fontsize=12)
    plt.savefig(os.path.join(save_path,sstr+".pdf"))

def blockSizeSpeedUp(normIdx=0,speedup=True):
    func = lambda currStr: True if ".csv" in currStr and currStr!="bestRuns.csv" else False
    files = list(filter(func,os.listdir(curr_path)))
    files.sort()
    curr_file = os.path.join(curr_path,files[normIdx])
    with open(curr_file,'r') as f:
        rdata = fileDisect(f)
    blocks,grids,ndata = zip(*rdata)
    ndata = np.asarray(ndata)
    blocks = list(set(blocks))
    grids = list(set(grids))
    blocks.sort()
    grids.sort()
    files.pop(normIdx) #remove normalization data
    ng = len(grids)
    nb = len(blocks)
    nf = len(files)
    norm_data = np.zeros((len(ndata),nf))
    for i,cfn in enumerate(files):
        curr_file = os.path.join(curr_path,cfn)
        with open(curr_file,'r') as f:
            rdata = fileDisect(f)
            cdata = ndata/np.asarray(list(zip(*rdata))[-1]) if speedup else np.asarray(list(zip(*rdata))[-1])/ndata
            norm_data[:,i] = cdata[:]
    average = np.zeros((ng,nf))
    for blk,i in enumerate(range(0,int(nb*ng),ng)):
        dataSet = norm_data[i:i+ng,:]
        average += dataSet
        generateFigure(grids,dataSet,blocks[blk],speedup)
    average /= nb #taking average
    generateFigure(grids,average,"average",speedup)

if __name__ == "__main__":
    speedup=True
    bestRunsSpeedUp(speedup=speedup)
    blockSizeSpeedUp(speedup=speedup)
