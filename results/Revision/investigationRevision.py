# @Author: Anthony Walker <sokato>
# @Date:   2020-04-17T17:31:11-07:00
# @Email:  walkanth@oregonstate.edu
# @Filename: revisionPlot.py
# @Last modified by:   sokato
# @Last modified time: 2020-04-20T17:15:58-07:00


import os
import matplotlib.pyplot as plt
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
save_path = os.path.join(curr_path,"figs")

def bestRunsSpeedUp(speedup=True):
    """
    bestRuns.csv: nX,classicFlat,classicLength,sweptFlat,sweptLength
    """
    curr_file = os.path.join(curr_path,os.path.join('investigations','bestRuns.csv'))
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
    files = list(filter(func,os.listdir(os.path.join(curr_path,'investigations'))))
    files.sort()
    curr_file = os.path.join(os.path.join(curr_path,'investigations'),files[normIdx])
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
        curr_file = os.path.join(curr_path,os.path.join('investigations',cfn))
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

def bestRunsComparison():
    """
    bestRuns.csv: nX,classicFlat,classicLength,sweptFlat,sweptLength
    """
    curr_file = os.path.join(curr_path,os.path.join('investigations','bestRuns.csv'))
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
    data = np.asarray(data)
    size = data[0,:]
    #This data
    data = data[1:,:]
    SVC(data,size)
    FVL(data,size)
    SLCF(data,size)

def SLCF(data,size):
    """Cross combinations - just a check"""
    lengthening = data[0,:]/data[3,:]
    flattening = data[1,:]/data[2,:]
    #plotting
    fig = plt.figure()
    # plt.set_title()
    plt.grid(b=True)
    plt.ylim(0,3)
    plt.xlabel("Grid Size",fontsize=14)
    plt.ylabel("Speedup",fontsize=14)
    plt.semilogx(size,lengthening,color="orange",linestyle="dashed",linewidth=3)
    plt.semilogx(size,flattening,color="steelblue",linestyle="dotted",linewidth=3)
    # idx = int(len(size)/2.75)
    # plt.text(size[idx],  np.mean(lengthening)+0.2, "Lengthening", fontsize=14) #, $S_{swept}=\\frac{t_{len,swept}}{t_{len,classic}}$
    # plt.text(size[idx],  np.mean(flattening)+0.05, "Flattening", fontsize=14) #, $S_{swept}=\\frac{t_{flat,swept}}{t_{flat,classic}}$
    plt.savefig(os.path.join(save_path,"slcf.pdf"))

def SVC(data,size):
    """This function takes the swept vs classic for a given strategies (e.g. lengthening)
    and produces S = t_len_swept/t_len_classic
    """
    lengthening = data[1,:]/data[3,:]
    flattening = data[0,:]/data[2,:]
    #plotting
    fig = plt.figure()
    # plt.set_title()
    plt.grid(b=True)
    plt.ylim(0,2)
    plt.xlabel("Grid Size",fontsize=14)
    plt.ylabel("Speedup",fontsize=14)
    plt.semilogx(size,lengthening,color="orange",linestyle="dashed",linewidth=3)
    plt.semilogx(size,flattening,color="steelblue",linestyle="dotted",linewidth=3)
    idx = int(len(size)/2.75)
    plt.text(size[idx],  np.mean(lengthening)+0.2, "Lengthening", fontsize=14) #, $S_{swept}=\\frac{t_{len,swept}}{t_{len,classic}}$
    plt.text(size[idx],  np.mean(flattening)+0.05, "Flattening", fontsize=14) #, $S_{swept}=\\frac{t_{flat,swept}}{t_{flat,classic}}$
    plt.savefig(os.path.join(save_path,"svc.pdf"))
    # plt.show()

def FVL(data,size):
    """This function takes the compares strategies for a given scheme (e.g. swept)
    and produces S = t_len_swept/t_flat_swept
    """
    classic = data[1,:]/data[0,:]
    swept = data[3,:]/data[2,:]
    #plotting
    fig = plt.figure()
    # plt.set_title()
    plt.grid(b=True)
    plt.ylim(0,4)
    plt.xlabel("Grid Size",fontsize=14)
    plt.ylabel("Speedup",fontsize=14)
    plt.semilogx(size,classic,color="blueviolet",linestyle="dashdot",linewidth=3)
    plt.semilogx(size,swept,color="deepskyblue",linestyle="solid",linewidth=3)
    idx = int(len(size)/2)
    plt.text(size[idx], np.mean(classic)+0.6, "Classic", fontsize=14) #, $S_{flat}=\\frac{t_{len,classic}}{t_{flat,classic}}$
    plt.text(size[idx], np.mean(swept)+0.2, "Swept", fontsize=14) #, $S_{flat}=\\frac{t_{len,swept}}{t_{flat,swept}}$
    plt.savefig(os.path.join(save_path,"fvl.pdf"))
    # plt.show()

if __name__ == "__main__":
    speedup=True
    bestRunsComparison()
    bestRunsSpeedUp(speedup=speedup)
    blockSizeSpeedUp(speedup=speedup)
