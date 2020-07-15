
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
import numpy as np
import os, csv, operator, itertools
import scipy.stats
import scipy.optimize

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
    # ax.grid(color='k', linewidth=1)
    ax.yaxis.labelpad = 0.5
    ax.xaxis.labelpad = 0.5

    if markbest:
        x,y = np.where(Z==np.amax(Z))
        ax.plot(uBlocks[x],uShares[y],linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)

    if markworst:
        x,y = np.where(Z==np.amin(Z))
        ax.plot(uBlocks[x],uShares[y],linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)

def outsideLabels(axes,fig):
    """Use this function to plot labels appropriately."""
    fig.subplots_adjust(wspace=0.2,hspace=0.3,right=0.8)
    labelBools = [(0,1),(0,0,),(1,1),(1,0)]
    for i,ax in enumerate(axes):
        if labelBools[i][1]:
            ax.set_ylabel('GPU Work Factor')
        #     ax.set_yticks(np.linspace(0,100,5))
        # else:
        #     ax.set_yticks([])
        if labelBools[i][0]:
            ax.set_xlabel('threads-per-block')
        #     ax.set_xticks([64, 256, 512, 768, 1024])
        # else:
        #     ax.set_xticks([])

        ax.set_yticks(np.linspace(0,100,5))
        ax.set_xticks([64, 256, 512, 768, 1024])

def rawMain(maxVList,fList,arraySize=[5e5,1e6,5e6,1e7]):
    #Find appropriate index
    for ctr,f in enumerate(fList):
        keys,mStruct,uGrids,uBlocks,uShares = createMainStruct(os.path.join("./splice-data",f))
        S,B = np.meshgrid(uShares,uBlocks)
        #Plots - figure and axes
        fig, axes = plt.subplots(ncols=2,nrows=2)
        axes = np.reshape(axes,(4,))
        maxsMins = list()
        Z = list()
        for i,size in enumerate(arraySize):
            arrIdxs = [1,2,4,6]
            arrayIdx = arrIdxs[i]
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
        outsideLabels(axes,fig)
        for i in range(len(arraySize)):
            performancePlot(axes[i],B,S,Z[i],minV,maxV,uBlocks,uShares,arraySize[i],markbest=True,markworst=True,ccm=cm.inferno_r,mbc=('k','w'))
        plt.savefig(os.path.join("./figs",f.split(".")[0]+".pdf"))

def eulerMain(arraySize=[5e5,1e6,5e6,1e7]):
    keys,eStructS,uGrids,uBlocks,uShares = createMainStruct('./splice-data/tEulerS-full.csv')
    eStructC = createMainStruct('./splice-data/tEulerC-full.csv',RG=False)
    #Plots - figure and axes
    fig, axes = plt.subplots(ncols=2,nrows=2)
    axes = np.reshape(axes,(4,))
    #Find appropriate index
    B = list()
    S = list()
    Z = list()
    maxsMins = list()
    for i,size in enumerate(arraySize):
        arrIdxs = [1,2,4,6]
        arrayIdx = arrIdxs[i]
        pData = getPerformanceData(arrayIdx,eStructC,eStructS,uBlocks,uShares)
        B.append(pData[0])
        S.append(pData[1])
        Z.append(pData[2])
        maxsMins.append(np.amax(Z[i]))
        maxsMins.append(np.amin(Z[i]))
    print("Euler")
    print(np.amax(maxsMins))
    print(np.amin(maxsMins))
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
    outsideLabels(axes,fig)
    for i in range(len(arraySize)):
        performancePlot(axes[i],B[i],S[i],Z[i],minV,maxV,uBlocks,uShares,arraySize[i],markbest=True,markworst=True)
    plt.savefig(os.path.join("./figs","eulerPerformance.pdf"))

def heatMain(arraySize=[5e5,1e6,5e6,1e7]):
    keys,eStructS,uGrids,uBlocks,uShares = createMainStruct('./splice-data/tHeatS-full.csv')
    eStructC = createMainStruct('./splice-data/tHeatC-full.csv',RG=False)
    #Plots - figure and axes
    fig, axes = plt.subplots(ncols=2,nrows=2)
    axes = np.reshape(axes,(4,))
    #Find appropriate index
    B = list()
    S = list()
    Z = list()
    maxsMins = list()
    for i,size in enumerate(arraySize):
        arrIdxs = [1,2,4,6]
        arrayIdx = arrIdxs[i]
        pData = getPerformanceData(arrayIdx,eStructC,eStructS,uBlocks,uShares)
        B.append(pData[0])
        S.append(pData[1])
        Z.append(pData[2])
        maxsMins.append(np.amax(Z[i]))
        maxsMins.append(np.amin(Z[i]))
    print("Heat")
    print(np.amax(maxsMins))
    print(np.amin(maxsMins))
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
    outsideLabels(axes,fig)
    for i in range(len(arraySize)):
        performancePlot(axes[i],B[i],S[i],Z[i],minV,maxV,uBlocks,uShares,arraySize[i],markbest=True,markworst=True)
    plt.savefig(os.path.join("./figs","heatPerformance.pdf"))

def checkMaxTimes():
    pth = "./splice-data"
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

def createBestConfigs(fList):
    """This creates best config files"""
    main = {}
    best = {}
    for f in fList:
        with open(os.path.join("./splice-data",f),"r") as file:
            reader = csv.reader(file)
            cline = next(reader)
            main[f] = [[float(item) for item in cline] for cline in reader]
        file.close()
    #Getting best configs
    with open('best-configs.csv','w') as cfile:
        cfile.write("model, tpb, share, grid, time\n")
        for f in fList:
            clist = main[f]
            clist.sort(key=operator.itemgetter(2,3))
            # print(f+", "+", ".join([str(curr) for curr in clist[0]])+"\\n")
            best[f] = [(clist[0][2],clist[0][3]),]
            cfile.write(f+", "+", ".join([str(curr) for curr in clist[0]])+"\n")
            for i in range(1,len(clist)):
                if clist[i][2] != clist[i-1][2]:
                    cfile.write(f+", "+", ".join([str(curr) for curr in clist[i]])+"\n")
                    best[f].append((clist[i][2],clist[i][3]))

    x1,y1 = zip(*best[fList[0]])
    x2,y2 = zip(*best[fList[1]])
    plotConfig("./figs/heat-best.pdf",x1,y1,x2,y2)
    ys = np.asarray(y1)/np.asarray(y2)
    print("Heat")
    print(np.amax(ys))
    print(np.amin(ys))
    speedUpPlot("./figs/speed-heat-best.pdf",x1,ys)
    results1 = scipy.stats.linregress(np.log(x2),np.log(y2))
    params1,covariance1 = scipy.optimize.curve_fit(powerlaw,x2,y2)
    x3,y3 = zip(*best[fList[2]])
    x4,y4 = zip(*best[fList[3]])
    plotConfig("./figs/euler-best.pdf",x3,y3,x4,y4)
    ys = np.asarray(y3)/np.asarray(y4)
    print("Euler")
    print(np.amax(ys))
    print(np.amin(ys))
    speedUpPlot("./figs/speed-euler-best.pdf",x3,ys)
    results2 = scipy.stats.linregress(np.log(x4),np.log(y4))
    params2,covariance2 = scipy.optimize.curve_fit(powerlaw,x4,y4)
    eqns = ["Heat","Euler"]
    for i,result in enumerate([results1,results2]):
        print("{}: R^2:{:f}, slope:{:f},intercept:{:f}".format(eqns[i],result.rvalue,result.slope,np.exp(result.intercept)))
    for i,param in enumerate([params1,params2]):
        print("{}: slope:{:f},intercept:{:f}".format(eqns[i],param[1],param[0]))
def powerlaw(x, a, b):
    return a * x**b

def plotConfig(save,x1,y1,x2=None,y2=None,labels=None):
    plt.close('all') #close other plots
    fig, ax = plt.subplots(ncols=1,nrows=1)
    ax.set_xlim([8e4,1e7])
    line1 = ax.loglog(x1,y1,marker='o',linestyle="--",color='#d95f02')
    ax.set_ylabel('time per timestep $[\\mu s]$')
    ax.set_xlabel('grid size')
    ax.grid()
    if x2 is not None:
        line2 = ax.loglog(x2,y2,marker='o',linestyle="solid",color='#7570b3')
        ax.legend(["Classic","Swept"],loc='upper left')
    plt.savefig(save)

def speedUpPlot(save,x,y):
    """Use this function to make speed up plots"""
    plt.close('all') #close other plots
    fig, ax = plt.subplots(ncols=1,nrows=1)
    ax.set_xlim([8e4,1e7])
    line1 = ax.semilogx(x,y,marker='o',linestyle="solid",color='#d95f02')
    ax.set_ylabel('speedup')
    ax.set_xlabel('grid size')
    ax.grid()
    plt.savefig(save)

def fileSplice(f1, f2):
    """Use this function to splice data files together."""
    with open(f1,'r') as fileOne:
        readFileOne = csv.reader(fileOne)
        listOne = [item for item in readFileOne]
        fileOne.close()
    tempList = []
    for i in range(3,len(listOne),5):
        tempList+=listOne[i:i+3]
    listOne=tempList
    with open(f2,'r') as fileTwo:
        readFileTwo = csv.reader(fileTwo)
        listTwo = [item for item in readFileTwo]
        fileTwo.close()
    header = listTwo[0]
    totalList = listOne+listTwo[1:]
    for ctr,item in enumerate(totalList):
        if '' in item:
            totalList[ctr]=[int(item[0]),0,float(item[2]),float(item[3])]
        else:
            totalList[ctr]=[int(item[0]),int(item[1]),float(item[2]),float(item[3])]
    totalList.sort(key=operator.itemgetter(0,1,2))
    name = f2.split("/")[-1]
    name = name.split(".")[0]
    with open("./splice-data/{}-full.csv".format(name),'w') as file:
        file.write(",".join(header)+"\n")
        for item in totalList:
            file.write(",".join([str(n) for n in item])+"\n")
        file.close()

def convertToShare(fList):
    """Use this function to convert data to share vs work factor."""
    squad = lambda b,c: (-b+np.sqrt(b*b-4*c))/2
    for f in fList:
        with open(os.path.join("./splice-data",f),"r") as file:
            reader = csv.reader(file)
            cline = next(reader)
            data = [[float(subitem) for subitem in item] for item in reader]
        data = list(zip(*data))
        Wt = -1*np.asarray(data[2])
        G = np.asarray(data[1])
        S = G/(1+G)
        for i,j in zip(S,G):
            print(i,j)

if __name__ == "__main__":
    maxTimes,fList = checkMaxTimes()
    fileSplice("./rawdata/tHeatC-fill.csv","./rawdata/tHeatC.csv")
    fileSplice("./rawdata/tHeatS-fill.csv","./rawdata/tHeatS.csv")
    fileSplice("./rawdata/tEulerS-fill.csv","./rawdata/tEulerS.csv")
    fileSplice("./rawdata/tEulerC-fill.csv","./rawdata/tEulerC.csv")
    eulerMain()
    heatMain()
    rawMain(maxTimes,fList)
    createBestConfigs(fList)
