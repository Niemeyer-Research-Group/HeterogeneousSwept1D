
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
import numpy as np
import itertools
import os,csv

def createMainStruct(fname,RG=True):
    f = open(fname,'r')
    line = f.readline()
    line = f.readline()
    data = []
    while line:
        line = line.replace(",,",",0,")
        data.append([float(x) for x in line.split(",")])
        line = f.readline()
    blocks,shares,grids,times = zip(*data)
    uniqBlocks = list(set(blocks))
    uniqShares = list(set(shares))
    uniqGrids = list(set(grids[:14]))
    uniqBlocks.sort()
    uniqShares.sort()
    uniqGrids.sort()
    keys = list(itertools.product(uniqBlocks,uniqShares))
    mainStruct = {key:[] for key in keys}
    for i in range(len(times)):
        mainStruct[(blocks[i],shares[i])].append(times[i])
    f.close()
    if RG:
        return keys,mainStruct,np.asarray(uniqGrids),np.asarray(uniqBlocks),np.asarray(uniqShares)
    else:
        return mainStruct

def testModify(fname,outname):
    """Use this function to modify the results in tEulerS and write them to a different file."""
    f = open(fname,'r')
    firstline = f.readline()
    line = f.readline()
    data = []
    while line:
        line = line.replace(",,",",0,")
        line = line.split(",")
        randN = np.random.uniform(low=0.5, high=5, size=(1,))[0]
        line[-1]=str(float(line[-1])/randN)
        data.append(",".join(line))
        line = f.readline()
    f.close()
    #Output results
    f = open(outname,'w')
    f.write(firstline)
    for d in data:
        f.write(d+"\n")
    f.close()

def getRawData(aIdx,struct,uBlocks,uShares):
    Z = np.zeros((len(uBlocks),len(uShares)))
    for i,blk in enumerate(uBlocks):
        for j,share in enumerate(uShares):
            key = (blk,share)
            Z[i,j] = struct[key][aIdx]
    return Z

def getPerformanceData(aIdx,struct1,struct2,uBlocks,uShares):
    Z = np.zeros((len(uBlocks),len(uShares)))
    S,B = np.meshgrid(uShares,uBlocks)
    for i,blk in enumerate(uBlocks):
        for j,share in enumerate(uShares):
            key = (blk,share)
            Z[i,j] = struct1[key][aIdx]/struct2[key][aIdx]
    return B,S,Z

def performancePlot(ax,B,S,Z,minV,maxV,uBlocks,uShares,ArrSize,ccm = cm.inferno,markbest=False,markworst=False,mbc=('w','k')):
    ax.contourf(B,S,Z,cmap=ccm)#,vmin=minV,vmax=maxV)
    nstr = '{:0.0f}'.format(ArrSize)
    ax.set_title('Grid Size: ${}\\times10^{}$'.format(nstr[0],len(nstr)-1))
    ax.set_ylabel('GPU share')
    ax.set_xlabel('threads-per-block')
    ax.set_xticks([64, 256, 512, 768, 1024])
    ax.set_yticks(np.linspace(0,100,5))
    # ax.grid(color='k', linewidth=1)
    ax.yaxis.labelpad = 0.5
    ax.xaxis.labelpad = 0.5
    if markbest:
        x,y = np.where(Z==np.amax(Z))
        ax.plot(uBlocks[x],uShares[y],linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)
    if markworst:
        x,y = np.where(Z==np.amin(Z))
        ax.plot(uBlocks[x],uShares[y],linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)

def rawMain(maxVList,fList,arraySize=[5e5,1e6,5e6,1e7]):
    #Find appropriate index
    for ctr,f in enumerate(fList):
        keys,mStruct,uGrids,uBlocks,uShares = createMainStruct(os.path.join("./rawdata",f))
        S,B = np.meshgrid(uShares,uBlocks)
        #Plots - figure and axes
        fig, axes = plt.subplots(ncols=2,nrows=2)
        fig.subplots_adjust(wspace=0.3,hspace=0.4,right=0.8)
        axes = np.reshape(axes,(4,))
        maxsMins = list()
        Z = list()
        for i,size in enumerate(arraySize):
            arrayIdx = i
            Z.append(getRawData(arrayIdx,mStruct,uBlocks,uShares))
        maxV = maxVList[ctr]
        minV = 0
        cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
        cbounds = np.linspace(minV,maxV,100)
        cbs = np.linspace(minV,maxV,4)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno_r),cax=cbar_ax,boundaries=cbounds)
        cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbs]])
        tick_locator = ticker.MaxNLocator(nbins=len(cbs))
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar_ax.set_title('Time [$\\mu s$]',y=1.01)
        #Make plots
        for i in range(len(arraySize)):
            performancePlot(axes[i],B,S,Z[i],minV,maxV,uBlocks,uShares,arraySize[i],markbest=True,markworst=True,ccm=cm.inferno_r,mbc=('k','w'))
        plt.savefig(os.path.join("./figs",f.split(".")[0]+".pdf"))

def eulerMain(arraySize=[5e5,1e6,5e6,1e7]):
    keys,eStructS,uGrids,uBlocks,uShares = createMainStruct('./rawdata/tEulerS.csv')
    eStructC = createMainStruct('./rawdata/tEulerC.csv',RG=False)
    #Plots - figure and axes
    fig, axes = plt.subplots(ncols=2,nrows=2)
    fig.subplots_adjust(wspace=0.3,hspace=0.4,right=0.8)
    axes = np.reshape(axes,(4,))
    #Find appropriate index
    B = list()
    S = list()
    Z = list()
    maxsMins = list()
    for i,size in enumerate(arraySize):
        arrayIdx = i
        pData = getPerformanceData(arrayIdx,eStructC,eStructS,uBlocks,uShares)
        B.append(pData[0])
        S.append(pData[1])
        Z.append(pData[2])
        maxsMins.append(np.amax(Z[i]))
        maxsMins.append(np.amin(Z[i]))
    maxV = np.ceil(np.amax(maxsMins))
    minV = np.floor(np.amin(maxsMins))
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    cbounds = np.linspace(minV,maxV,100)
    cbs = np.linspace(minV,maxV,5)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),cax=cbar_ax,boundaries=cbounds)
    cbar.ax.set_yticklabels(["{:0.1f}".format(i) for i in cbs])
    tick_locator = ticker.MaxNLocator(nbins=len(cbs))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title('Speed up',y=1.01)
    #Make plots
    for i in range(len(arraySize)):
        performancePlot(axes[i],B[i],S[i],Z[i],minV,maxV,uBlocks,uShares,arraySize[i],markbest=True,markworst=True)
    plt.savefig(os.path.join("./figs","eulerPerformance.pdf"))

def heatMain(arraySize=[5e5,1e6,5e6,1e7]):
    keys,eStructS,uGrids,uBlocks,uShares = createMainStruct('./rawdata/tHeatS.csv')
    eStructC = createMainStruct('./rawdata/tHeatC.csv',RG=False)
    #Plots - figure and axes
    fig, axes = plt.subplots(ncols=2,nrows=2)
    fig.subplots_adjust(wspace=0.3,hspace=0.4,right=0.8)
    axes = np.reshape(axes,(4,))
    #Find appropriate index
    B = list()
    S = list()
    Z = list()
    maxsMins = list()
    for i,size in enumerate(arraySize):
        arrayIdx = i
        pData = getPerformanceData(arrayIdx,eStructC,eStructS,uBlocks,uShares)
        B.append(pData[0])
        S.append(pData[1])
        Z.append(pData[2])
        maxsMins.append(np.amax(Z[i]))
        maxsMins.append(np.amin(Z[i]))
    maxV = np.ceil(np.amax(maxsMins))
    minV = np.floor(np.amin(maxsMins))
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    cbounds = np.linspace(minV,maxV,100)
    cbs = np.linspace(minV,maxV,5)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),cax=cbar_ax,boundaries=cbounds)
    cbar.ax.set_yticklabels(["{:0.1f}".format(i) for i in cbs])
    tick_locator = ticker.MaxNLocator(nbins=len(cbs))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title('Speed up',y=1.01)
    #Make plots
    for i in range(len(arraySize)):
        performancePlot(axes[i],B[i],S[i],Z[i],minV,maxV,uBlocks,uShares,arraySize[i],markbest=True,markworst=True)
    plt.savefig(os.path.join("./figs","heatPerformance.pdf"))

def checkMaxTimes():
    pth = "./rawdata"
    maxTimes = []
    fList = os.listdir(pth)
    for f in fList:
        cf = os.path.join(pth,f)
        with open(cf,'r') as file:
            line = file.readline()
            line = file.readline()
            data = []
            while line:
                line = line.replace(",,",",0,")
                data.append(float(line.split(",")[-1]))
                line = file.readline()
            file.close()
            l = len(str(int(np.amax(data))))
            maxTimes.append(round(np.amax(data),-(l-2)))
    return maxTimes,fList

if __name__ == "__main__":
    pth = "./rawdata"
    eulerMain()
    heatMain()
    maxTimes,fList = checkMaxTimes()
    rawMain(maxTimes,fList)
